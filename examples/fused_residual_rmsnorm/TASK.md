I want the pre-norm residual path of my model to be a single fused kernel. The function is
`add_rmsnorm` in `minilm/model.py`; every block calls it twice and the final norm once. It adds
the sublayer output into the fp32 residual stream and RMS-normalises the result for the next
sublayer, returning both the new stream and the normalised bf16 view.

Training config is `minilm/config.py` (`TrainConfig`, `ModelConfig`): bf16 autocast, batch 8,
sequence 512, `d_model` 1024, 4 layers. The smoke run is `python train_smoke.py --steps 50`; it prints `median_step_ms` and `final_loss`.

Please take it all the way: spec, kernel with a fused backward, verification and benchmark
against speed of light, and integrate it into `minilm/model.py` behind a flag so I can switch
back to the eager code. I care most about the shapes the config actually uses.
