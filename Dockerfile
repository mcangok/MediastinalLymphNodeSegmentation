FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/app /input /output/images/mediastinal-lymph-node-segmentation/ \
    && chown -R user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user environment_totalsegmentator.yml /opt/app/
COPY --chown=user:user environment_nnunetv2.yml /opt/app/
COPY --chown=user:user requirements.txt /opt/app/

# local env for postprocessing
RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
RUN python -m pip install -r requirements.txt

# env for totalsegmentator (nnUNetv1) and nnUNetv2 which are incompatible
RUN conda env create -f /opt/app/environment_totalsegmentator.yml
RUN conda env create -f /opt/app/environment_nnunetv2.yml

RUN conda run -n totalsegmentator "totalseg_download_weights"
COPY --chown=user:user process.py /opt/app/

# nnunetv2 LNQ segmentation docker code
COPY --chown=algorithm:algorithm lnq_segmentation /opt/algorithm

ENTRYPOINT [ "python", "-m", "process" ]



