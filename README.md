# AIMO 3 template code

This is some template code where you can use to serve vLLM and solve [AI Mathematical Olympaid 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3)


# Setup

Run `uv sync` to install packages in .venv


# Deploy inference

```
uv run modal deploy inference.py
```

When deployed, you can visit to see available models

```
https://<handle>--example-gpt-oss-inference-serve.modal.run/v1/models
```


You will need to save the url to `env.json`. It looks something like

```
{
  "REMOTE_VLLM_URL": "https://<handle>--example-gpt-oss-inference-serve.modal.run/v1"
}
```


# Solve problems

```
uv run python3 kaggle.py
```

You should see problems being solve.

The same code could be used on Kaggle.

How to upload to Kaggle
- File > Editor Type > Script
- Paste `kaggle.py` over
- File > Editor Type > Notebook

To run with H100 GPU (which you make make a submission with)
- Session options > Accelerator > GPU H100
- Internet off

To run without H100 GPU
- Add-ons > Secrets > Add Secret > Label: REMOTE_VLLM_URL, Value: (copy the URL over)
- Session options > Accelerator > None
- Internet on
