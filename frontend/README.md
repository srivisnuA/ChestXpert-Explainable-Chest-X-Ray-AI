# ChestXpert Demo UI

The demonstration interface uses Streamlit.

Run from the repository root:

```bash
streamlit run frontend/app.py
```

The interface supports:

- X-ray upload
- Model checkpoint selection
- Configurable flagging threshold
- Per-finding probability display
- Flagged-finding summary
- Research/clinical-review disclaimer

The next UI iteration will connect the Grad-CAM module to display a class-specific visual explanation.

This interface is a research/educational demonstration and not a diagnostic tool.
