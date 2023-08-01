# Dummy Data, Processor, and Model Generation

The scripts here generate a dummy HuggingFace dataset, ViTImageProcessor, and ViTForImageClassification model to use in tests. The parameters of the generated files are set in `dummy_config.json`. To generate the files run the three scripts:

```bash
python make_dummy_data.py
python make_dummy_model.py
python make_dummy_processor.py
```
