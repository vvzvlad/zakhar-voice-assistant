# training/ — пайплайн обучения «Захар» (microWakeWord)

Реальные скрипты и конфиги, которыми обучалась модель [../model/zakhar.tflite](../model/zakhar.tflite).
Полное пошаговое описание процесса и порядок запуска — в
[../TRAINING_PROCESS.md](../TRAINING_PROCESS.md).

> Пути в скриптах **абсолютные** (`/home/claude/zakhar-mww/...`) — это копия рабочих файлов
> с обучающей ноды «как есть». Для запуска на другой машине поправь пути.

| Файл | Назначение |
|------|-----------|
| `dl_negatives.sh` | скачать предгенерированные негативы с HF `kahrendt/microwakeword` |
| `prep_backgrounds.sh` | подготовить фон для аугментации: FMA→16k wav + MIT RIR |
| `generate_synth.py` | синтетические позитивы «Захар» через Piper-TTS (рус. голоса) |
| `gen_speech_neg.py` | синтетические русские фразы-негативы (hard-negatives/пробы) |
| `build_features.py` | 40-полосные спектрограммы (RaggedMmap, step 10 мс) + аугментация |
| `training_parameters.yaml` | конфиг обучения v1 (отгружено) |
| `training_parameters_v2.yaml` | конфиг итерации v2 (neg_weight 30 — отклонена) |
| `train.sh` / `train_v2.sh` | запуск `model_train_eval.py` на CPU (mixednet, 25k шагов) |
| `evaluate.py` | оценка quantized streaming: recall на held-out + FAPH на ambient |
| `generate_manifest.py` | сборка JSON-манифеста v2 для ESPHome |
