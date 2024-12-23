# meta-llama/Llama-Guard-3-11B-Vision Cog model

This is an implementation of [meta-llama/Llama-Guard-3-11B-Vision](https://huggingface.co/meta-llama/Llama-Guard-3-11B-Vision) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

To run a safe user prediction:

    cog predict -i image=@bakery.jpg -i prompt="Which one should I buy?"

Output

    safe