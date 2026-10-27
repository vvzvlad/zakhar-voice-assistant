#!/bin/sh
# Plays the activation sound each time quasar spots the wake word: the firmware's own
# playStartVoiceInputEarcon stays silent, so this follows the spot line in quasar's own log
# (maind.log, onPhraseSpotted) and plays the board's stock earcon asset (spotter_start.opus)
# through maind --gst-launch, the player the station recipes use for local sounds.
LOG=/data/quasar/daemons_logs/maind.log
while true; do
	tail -F -n0 "$LOG" 2>/dev/null | while IFS= read -r line; do
		case "$line" in
		*onPhraseSpotted*)
			cd /system/vendor/quasar
			LD_LIBRARY_PATH=/system/vendor/quasar:libs ./maind --gst-launch \
				"filesrc location=/system/vendor/quasar/sounds/common/spotter_start.opus ! decodebin ! audioconvert ! audioresample ! alsasink" \
				>/dev/null 2>&1
			echo "$(date +%T) rc=$?" >>/tmp/alice-blip.log
			;;
		esac
	done
	sleep 2
done
