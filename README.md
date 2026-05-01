# AV-TS-ASR Parakeet

This repository contains the official implementation of our **CHiME-9 MCoRec AV-TS-ASR system** based on AV-Hubert and Nvidia Parakeet-0.6b v2.

## Environment Setup
1. Create a new conda environment: `conda create -n av_parakeet python=3.11 -y` and activate it using `conda activate av_parakeet`.

2. Install ffmpeg `conda install -c conda-forge "ffmpeg<8" -y`.

3. Install the python dependencies: `pip install -r requirements.txt`.

4. Download AV-Hubert model finetuned on MCoRec: `wget https://huggingface.co/MCoRecChallenge/MCoRec-baseline/resolve/main/model-bin.zip
unzip model-bin.zip; unzip model-bin.zip`

## Data Setup

**The following data setup is required if you want to train our models. If you want to use it for inference only, you can skip this section and continue to Inference section below.**

Our training codebase uses [Lhotse](https://github.com/lhotse-speech/lhotse) manifests. For inference, you can run our model on single video file, directory containing video files, or MCoRec data.

### MCoRec Data Setup
To prepare the filled-in speaker tracks and Lhotse manifests, run:
```bash
./scripts/data_prep/prepare_mcorec.sh {path_to_mcorec_dataset}
```

The path should point to a directory with `train` and `dev` subdirectories (i.e., MCoRec dataset root).

## Inference
We currently support two inference modes: MCoRec (CHiME-9) and standard per-video inference.

1. (Optional) Download the [MCoRec data from HuggingFace](https://huggingface.co/datasets/MCoRecChallenge/MCoRec).

2. Make sure you have access to: `BUT-FIT/AV-Parakeet_v0.1`.

3. If you want to infer MCoRec data, run the following inference command: 
    ```bash
    python infer_mcorec.py \
        +session_dir={path_to_mcorec_data}/dev/ \
        +output_dir=predictions \
        +timestamps=true \
        +mode=full 
    ```

4. If you want to infer arbitrary video/dictionary full of videos, run:
    ```bash
    python infer.py --input {path_to_dir}/{video}.mp4 --output-dir output_transcripts`
    ```

    or

    ```bash
    python infer.py --input "{path_to_dir}" --output-dir output_transcripts`
    ```

The output of `infer_mcorec.py` is the in [CHiME-9 MCoRec task format](https://www.chimechallenge.org/challenges/chime9/task1/submission).

The output `infer.py` is a directory with a single `ctm` file per video (`{output_directory}/{video_name}.ctm`).

## Training
The training is built on top of the [Nvidia NeMo](https://github.com/NVIDIA-NeMo/NeMo) toolkit. We recommend getting familiar with the basics, although it is not fully required.

We use WandB for logging by default, make sure you are locally logged in, or change the logging to tensorboard by setting `create_tensorboard_logger: true` and `create_wandb_logger: false` in `conf/av_parakeet.yaml`.

If you have changed any paths, go to `conf/av_parakeet.yaml` and change the particular values. Otherwise, you can keep it intact.

To run the training with the default settings, run:
```bash
python train.py +exp_dir="exps/"
```

It will automatically create `./exps/av_parakeet` directory with checkpoints.

## 📚 Citation
If you use our models or code, please cite the following works:
```
@misc{klement2026descriptionchime9mcorecchallenge,
      title={BUT System Description for CHiME-9 MCoRec Challenge}, 
      author={Dominik Klement and Alexander Polok and Nguyen Hai Phong and Prachi Singh and Lukáš Burget},
      year={2026},
      eprint={2604.27436},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2604.27436}, 
}
```