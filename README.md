# **When Intents take the Long Walk: Flexible Entity Resolution of Long Texts**

RepIPE (Representation-based Incremental Passage Election), an end-to-end solution that combines a novel representation algorithm, which provably converges, with a dedicated matching procedure.
![An illustration of RepIPE’s pipeline. Train anchor passages are used to learn data item representations, which are averaged within each equivalence class to form centroids. At inference time, these centroids support both match prediction and passage selection.](https://github.com/sigsub/RePipe/blob/main/pipeline_new.png)


# Repository Structure
* ### Main Inference Walkthrough
   * [res_run_through.ipynb](https://github.com/sigsub/RePipe/blob/main/res_run_through.ipynb)
* ### Transformer Aggregation Implementation and Inference
   * [transformer_agg.py](https://github.com/sigsub/RePipe/blob/main/transformer_agg.py)
   * [transformer_agg_pipeline](https://github.com/sigsub/RePipe/blob/main/transformer_agg_pipeline.ipynb)
* ### Data Collection and Processing 
   * [dataloader_and_prep.py](https://github.com/sigsub/RePipe/blob/main/dataloadrer_and_prep.py)
   * [download_scientific_papers.ipynb](https://github.com/sigsub/RePipe/blob/main/download_scientific_papers.ipynb)
   * [extracting_papers_html.py](https://github.com/sigsub/RePipe/blob/main/extracting_papers_html.py)
   * [html_parsing_utils.py](https://github.com/sigsub/RePipe/blob/main/html_parsing_utils.py)
   * [loading_paper_utils.py](https://github.com/sigsub/RePipe/blob/main/loading_paper_utils.py)
   * [news_data_scraping.py](https://github.com/sigsub/RePipe/blob/main/news_data_scraping.py)
   * [news_data_split_pipeline.ipynb](https://github.com/sigsub/RePipe/blob/main/news_data_split_pipeline.ipynb)
   * [generating_summary.py](https://github.com/sigsub/RePipe/blob/main/generating_summary.py)
* ### Representation Improvement Algorithm Implementation and Entity Representation Extraction
   * [data_refining_pipeline.py](https://github.com/sigsub/RePipe/blob/main/data_refining_pipeline.py)
* ### Misc.
   * [editing_fig_pkls.ipynb](https://github.com/sigsub/RePipe/blob/main/editing_fig_pkls.ipynb) — Visualizations for the paper
   * [matching_metrics.py](https://github.com/sigsub/RePipe/blob/main/matching_metrics.py) — Implementation of the Macro-F1 metric.
   * [finetune_hf_model_for_reps.py](https://github.com/sigsub/RePipe/blob/main/finetune_hf_model_for_reps.py) — Finetuning a huggingface SBERT model using a paired dataset.
   * [requirements.txt](https://github.com/sigsub/RePipe/blob/main/requirements.txt) — python environment requirements file.
