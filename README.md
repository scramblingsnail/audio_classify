# Audio Classification based on LSTM

## Feature Extraction
[mfcc](./src/feature_extract/audio_features.py) or [mel-spectrogram](./src/feature_extract/audio_features.py). 
The extracted data is recorded in the corresponding directory `mfcc` or `mel-spectrogram` under [data](./data)

## Model
[LSTM](./src/net/lstm.py)

## QuickStart
- Create configuration file in this [directory](./src/cfg) for your experiment, refer to [config1](./src/cfg/config1.yaml) as an example.
- Turn to [run](./run.py), change the parameter `config_flag` to the name of your configuration file.
- Run.