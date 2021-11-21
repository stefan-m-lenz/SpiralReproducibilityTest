using Pkg;
Pkg.activate(".")
include("SpiralExample.jl")

# Generate spiral samples
using Random;
Random.seed!(11);
# generate some training data
spiraldata = SpiralExample.spiral_samples(nspiral = 100,
      nsample = 30, start = 0.0, stop = 6*pi, a = 0.0, b = 1.0)

using Random;
Random.seed!(11);

# Define model architecture and initialize model
model = SpiralExample.LatentTimeSeriesVAE(latent_dim = 4, obs_dim = 2,
      rnn_nhidden = 25, f_nhidden = 20, dec_nhidden = 20)

function printmonitoring(epoch, loss)
   println("Epoch $epoch: $loss")
end

SpiralExample.train!(model, spiraldata.samp_trajs, spiraldata.samp_ts,
      epochs = 5, learningrate = 0.01, monitoring = printmonitoring)
