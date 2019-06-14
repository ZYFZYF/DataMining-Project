for((i=0;i<=4;i++));
do
python run_classifier.py --output_dir=./cv_${i}_output/ --train_file=train_${i}.tsv --eval_file=eval_${i}.tsv --task_name=mrpc --do_train=true --do_eval=true --do_predict=true --data_dir=data --vocab_file=model/vocab.txt --bert_config_file=model/bert_config.json --init_checkpoint=model/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=5e-5 --num_train_epochs=2.0
done