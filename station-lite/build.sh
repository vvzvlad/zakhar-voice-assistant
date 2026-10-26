#!/bin/bash
# Builds station-lite/build/zakhar-station and zakhar-replay: run by `make station-build` inside
# dockcross/linux-armv7l-musl, from the root of the tree tarred in, with the cache volume on /cache.
# Third-party code is fetched at pinned versions into /cache/src (checked against its sha256), and
# the two heavy libraries (TFLite Micro, webrtc-audio-processing with its abseil) are built once
# into /cache/lib-<key>, the key being their versions and the flags; everything else is compiled on
# every run. Usage: station-lite/build.sh <version for hello.fw>
set -euo pipefail

VERSION=$1
# Cortex-A53 cores running a 32-bit hard-float userland; the binary is static.
FLAGS="-mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard"
# Parallel compiles: firmware/CMakeLists.txt's cap, which keeps a build inside cirunner's 19 GB.
JOBS=24

CJSON_V=1.7.19
TFLM_V=1.3.3~1   # espressif/esp-tflite-micro, firmware/components/wake_word/idf_component.yml
MSF_V=1.2.3      # esphome/esp-micro-speech-features, the same file
WEBRTC_V=2.1     # webrtc-audio-processing, PulseAudio's packaging
TINYALSA_V=2.0.0

SRC=/cache/src
LIB=/cache/lib-$(echo "$CC $FLAGS $TFLM_V $WEBRTC_V" | sha256sum | cut -c1-12)
export OBJ=/tmp/obj
OUT=station-lite/build
mkdir -p "$SRC" "$LIB" "$OBJ" "$OUT"

# fetch <url> <sha256> <file in $SRC>
fetch() {
    if [ ! -f "$SRC/$3" ]; then
        curl -fsSL -o "$SRC/$3.part" "$1"
        mv "$SRC/$3.part" "$SRC/$3"
    fi
    echo "$2  $SRC/$3" | sha256sum -c --quiet
}

# unpack <archive in $SRC> <directory in $SRC it makes or is unpacked into>
unpack() {
    [ -d "$SRC/$2" ] && return
    case "$1" in
    *.zip) python3 -m zipfile -e "$SRC/$1" "$SRC/$2" ;;
    *) tar -C "$SRC" -xf "$SRC/$1" ;;
    esac
}

# compile <compiler and flags...> < sources: each source into $OBJ/<its path>.o, JOBS at a time.
compile() {
    xargs -P "$JOBS" -I{} bash -c 'o=$OBJ/${1#/}.o; mkdir -p "${o%/*}"; shift; exec "$@" -c "$0" -o "$o"' {} {} "$@"
}

# objects < sources: the objects compile() made of them.
objects() {
    sed "s|^/*|$OBJ/|; s|\$|.o|"
}

# ---------------------------------------------------------------- sources

fetch "https://raw.githubusercontent.com/DaveGamble/cJSON/v$CJSON_V/cJSON.c" \
    298581a04a36c0165da4b0aade235c23088cb2faa58651d720ea2f3706ed0b0d "cJSON-$CJSON_V.c"
fetch "https://raw.githubusercontent.com/DaveGamble/cJSON/v$CJSON_V/cJSON.h" \
    25b0145150d500498e4d209cec69c18c42cf818bffcc54690be3b895a2a16dee "cJSON-$CJSON_V.h"
mkdir -p "$SRC/cJSON-$CJSON_V"
cp "$SRC/cJSON-$CJSON_V.c" "$SRC/cJSON-$CJSON_V/cJSON.c"
cp "$SRC/cJSON-$CJSON_V.h" "$SRC/cJSON-$CJSON_V/cJSON.h"
fetch "https://components-file.espressif.com/components/espressif/esp-tflite-micro/$TFLM_V/espressif__esp-tflite-micro-v${TFLM_V/\~/_}.zip" \
    221b9ca49b236ad666dc453ee8dbeed1040ab41246784133e9f1d339b8d468da "esp-tflite-micro-$TFLM_V.zip"
unpack "esp-tflite-micro-$TFLM_V.zip" "esp-tflite-micro-$TFLM_V"
fetch "https://components-file.espressif.com/components/esphome/esp-micro-speech-features/$MSF_V/esphome__esp-micro-speech-features-v$MSF_V.zip" \
    052a8f32e7a82f49593bfdc067638a8cee6542443726d1a7e5b01b3bd9d67eac "esp-micro-speech-features-$MSF_V.zip"
unpack "esp-micro-speech-features-$MSF_V.zip" "esp-micro-speech-features-$MSF_V"
fetch "https://freedesktop.org/software/pulseaudio/webrtc-audio-processing/webrtc-audio-processing-$WEBRTC_V.tar.xz" \
    ae9302824b2038d394f10213cab05312c564a038434269f11dbf68f511f9f9fe "webrtc-audio-processing-$WEBRTC_V.tar.xz"
unpack "webrtc-audio-processing-$WEBRTC_V.tar.xz" "webrtc-audio-processing-$WEBRTC_V"
fetch "https://github.com/tinyalsa/tinyalsa/archive/refs/tags/v$TINYALSA_V.tar.gz" \
    573ae0b2d3480851c1d2a12503ead2beea27f92d44ed47b74b553ba947994ef1 "tinyalsa-$TINYALSA_V.tar.gz"
unpack "tinyalsa-$TINYALSA_V.tar.gz" "tinyalsa-$TINYALSA_V"

CJSON=$SRC/cJSON-$CJSON_V
TFLM=$SRC/esp-tflite-micro-$TFLM_V
MSF=$SRC/esp-micro-speech-features-$MSF_V
WEBRTC=$SRC/webrtc-audio-processing-$WEBRTC_V
TINYALSA=$SRC/tinyalsa-$TINYALSA_V

# ---------------------------------------------------------------- the cached libraries

# TFLite Micro as the component's CMakeLists.txt builds it, with the reference kernels in place of
# esp-nn's, and without the signal library and TFLM's own microfrontend: the wake-word engine uses
# esp-micro-speech-features' frontend.
TFLM_INC="-I$TFLM -I$TFLM/third_party/gemmlowp -I$TFLM/third_party/flatbuffers/include -I$TFLM/third_party/ruy
    -I$TFLM/third_party/kissfft"
TFLM_DEF="-DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON"
if [ ! -f "$LIB/libtflm.a" ]; then
    m=$TFLM/tensorflow/lite/micro
    l=$TFLM/tensorflow/lite
    printf '%s\n' \
        $m/{debug_log,flatbuffer_utils,memory_helpers,micro_allocation_info,micro_allocator,micro_context}.cc \
        $m/{micro_interpreter_context,micro_interpreter_graph,micro_interpreter,micro_log,micro_op_resolver}.cc \
        $m/{micro_profiler,micro_resource_variable,micro_time,micro_utils,recording_micro_allocator,system_setup}.cc \
        $m/tflite_bridge/*.cc $m/kernels/*.cc \
        $l/kernels/kernel_util.cc $m/memory_planner/{greedy,linear}_memory_planner.cc \
        $m/arena_allocator/{non_persistent,persistent}_arena_buffer_allocator.cc \
        $m/arena_allocator/{recording_single,single}_arena_buffer_allocator.cc \
        $l/core/c/common.cc $l/core/api/{flatbuffer_conversions,tensor_utils}.cc \
        $l/kernels/internal/{common,quantization_util,portable_tensor_utils,tensor_utils,tensor_ctypes}.cc \
        $l/kernels/internal/reference/{portable_tensor_utils,comparisons}.cc \
        $TFLM/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc \
        $TFLM/tensorflow/compiler/mlir/lite/schema/schema_utils.cc >"$OBJ/tflm.list"
    compile "$CXX" $FLAGS -O3 -std=gnu++17 -fno-rtti -fno-exceptions -fno-threadsafe-statics $TFLM_DEF $TFLM_INC \
        <"$OBJ/tflm.list"
    objects <"$OBJ/tflm.list" | xargs "$AR" rcs "$LIB/libtflm.a.part"
    mv "$LIB/libtflm.a.part" "$LIB/libtflm.a"
fi

# webrtc-audio-processing and the abseil it builds as a meson subproject, static.
if [ ! -f "$LIB/webrtc.stamp" ]; then
    cat >"$LIB/cross.ini" <<EOF
[binaries]
c = '$CC'
cpp = '$CXX'
ar = '$AR'
strip = '$CROSS_ROOT/bin/$CROSS_TRIPLE-strip'

[built-in options]
c_args = [$(printf "'%s', " $FLAGS)]
cpp_args = [$(printf "'%s', " $FLAGS)]

[host_machine]
system = 'linux'
cpu_family = 'arm'
cpu = 'cortex-a53'
endian = 'little'
EOF
    rm -rf "$LIB/webrtc-build"
    meson setup "$LIB/webrtc-build" "$WEBRTC" --cross-file "$LIB/cross.ini" --prefix "$LIB/webrtc" \
        --buildtype release -Ddefault_library=static --wrap-mode forcefallback
    meson compile -C "$LIB/webrtc-build" -j "$JOBS"
    meson install -C "$LIB/webrtc-build"
    touch "$LIB/webrtc.stamp"
fi
WEBRTC_LIBS=$(find "$LIB/webrtc-build" -name '*.a' | sort)

# ---------------------------------------------------------------- the binaries

COMPAT=station-lite/compat
FW=firmware
INC="-I$COMPAT -Istation-lite/main -I$FW/components/zakhar_link/include -I$FW/components/zakhar_link
    -I$FW/components/wake_word/include -I$FW/main -I$CJSON -I$TINYALSA/include -I$MSF/include"
CFLAGS="$FLAGS -O2 -std=gnu17 -Wall -Wextra $INC"
CXXFLAGS="$FLAGS -O2 -std=gnu++17 -Wall -Wextra $INC"

# tinyalsa as its CMakeLists.txt builds the library, plugins off.
printf '%s\n' "$TINYALSA"/src/{pcm,pcm_hw,pcm_plugin,snd_card_plugin,mixer,mixer_hw,mixer_plugin}.c >"$OBJ/tinyalsa.list"
compile "$CC" $FLAGS -O2 -std=c99 -D_POSIX_C_SOURCE=200809L -I"$TINYALSA/include" <"$OBJ/tinyalsa.list"
# esp-micro-speech-features as its CMakeLists.txt builds it; heap_caps from the compat layer.
ls "$MSF"/src/*.c >"$OBJ/msf.list"
compile "$CC" $FLAGS -O2 -std=gnu17 -I"$MSF/include" -I"$MSF/src" -I"$COMPAT" <"$OBJ/msf.list"

printf '%s\n' "$CJSON/cJSON.c" \
    $COMPAT/{esp,freertos,mdns,mdns_packet,websocket,ws_frame}.c \
    $FW/components/zakhar_link/{zakhar_link,zl_core}.c $FW/main/{audio_core,ui_core}.c \
    station-lite/main/{main,audio,mic_chain,leds,volume}.c >"$OBJ/c.list"
compile "$CC" $CFLAGS -DSTATION_VERSION="\"$VERSION\"" <"$OBJ/c.list"
compile "$CC" $CFLAGS <<<"station-lite/tools/replay.c"
compile "$CXX" $CXXFLAGS $TFLM_DEF $TFLM_INC <<<"$FW/components/wake_word/wake_word.cpp"
# The installed abseil headers sit in include/, which the .pc of a build with abseil as a
# subproject does not name.
compile "$CXX" $CXXFLAGS $(PKG_CONFIG_PATH="$LIB/webrtc/lib/pkgconfig" pkg-config --cflags webrtc-audio-processing-2) \
    -I"$LIB/webrtc/include" <<<"station-lite/main/apm.cc"

cat "$OBJ"/{c,tinyalsa,msf}.list <(echo "$FW/components/wake_word/wake_word.cpp") <(echo station-lite/main/apm.cc) \
    | objects >"$OBJ/link.list"
"$CXX" $FLAGS -static -no-pie -o "$OUT/zakhar-station" $(cat "$OBJ/link.list") \
    -Wl,--start-group "$LIB/libtflm.a" $WEBRTC_LIBS -Wl,--end-group -lm
# The replay tool: the capture processing and the wake-word engine with its log, nothing of the
# sound card or the link.
cat <(printf '%s\n' station-lite/tools/replay.c station-lite/main/{mic_chain.c,apm.cc} $COMPAT/{esp,freertos}.c \
    "$FW/components/wake_word/wake_word.cpp") "$OBJ/msf.list" | objects >"$OBJ/replay.list"
"$CXX" $FLAGS -static -no-pie -o "$OUT/zakhar-replay" $(cat "$OBJ/replay.list") \
    -Wl,--start-group "$LIB/libtflm.a" $WEBRTC_LIBS -Wl,--end-group -lm
