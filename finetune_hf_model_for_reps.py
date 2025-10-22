
from datasets import load_dataset # Assuming you have your dataset object already
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, SentenceTransformer
from sentence_transformers.losses import ContrastiveLoss
from sentence_transformers.evaluation import BinaryClassificationEvaluator
import os

# Used for cuda setup (on our machine), omit if no need.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def finetune_hf_model_for_reps(model_name, dataset_csv_path, checkpoints_dir, final_model_save_dir, batch_size=32, lr=2e-5, n_epochs=5, prompts=None):
    model = SentenceTransformer(model_name).to("cuda")
    loss = ContrastiveLoss(model, margin = 1)
    dataset = load_dataset("csv", data_files=dataset_csv_path, split='train')
    dataset = dataset.map(remove_columns=['id1', 'id2'])
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    TRAIN_SET = train_test['train']
    EVAL_SET = train_test['test']
    EVAL_STEPS = int(len(TRAIN_SET) / batch_size / 10) #every 0.1 epoch
    training_args = SentenceTransformerTrainingArguments(
        output_dir=checkpoints_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        # prompts={'text1': 'query: ', 'text2': 'query: '},
        prompts = prompts,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,  
        save_strategy="steps",
        save_steps=EVAL_STEPS,  
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="cosine_f1",
        greater_is_better=True,
        logging_dir="logs",
        logging_steps=100,
    )
    dev_evaluator = BinaryClassificationEvaluator(
        sentences1=EVAL_SET["text1"],
        sentences2=EVAL_SET["text2"],
        labels=EVAL_SET["label"],
        name="eval",
    )
    # dev_evaluator = BinaryClassificationEvaluator()
    mets= dev_evaluator(model)
    print(mets)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=TRAIN_SET,
        loss=loss,
        evaluator=dev_evaluator,
    )
    # print(trainer._determine_best_metric(mets, trial=None))
    trainer.train()
    trainer.save_model(final_model_save_dir)


def main():
    # example usage
    data_path = 'YOUR/DATA/path'
    model_name = 'USED/HUGGINGFACE/MODEL/NAME'
    checkpoints_dir = 'CHECKPOINT/SAVE/DIRECTORY'
    final_model_save_dir = 'FINAL/MODEL/SAVE/DIR'
    finetune_hf_model_for_reps(model_name, data_path, checkpoints_dir, final_model_save_dir,n_epochs=5, batch_size=32, lr=2e-5, prompts={'text1': 'query: ', 'text2': 'query: '})

if __name__ == "__main__":
    main()



 