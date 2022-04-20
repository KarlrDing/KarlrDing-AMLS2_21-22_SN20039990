def model_initialization(MODEL = 'bert-base-uncased'):
    #model import
    import tensorflow as tf
    from transformers import TFAutoModelForSequenceClassification
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=5)
    return model

def model_train(model,model_name,tf_train_dataset,tf_validation_dataset,dataset_name,PARAMETERS):
    import tensorflow as tf
    from transformers import AdamWeightDecay
    from transformers import WarmUp
    #parameters
    EPOCH       = PARAMETERS[0]#6
    ILR         = PARAMETERS[1]#5e-5
    D_S         = PARAMETERS[2]#600
    D_R         = PARAMETERS[3]#0.9
    W_S         = PARAMETERS[4]#200

    #define callback function
    train_los = []
    train_acc = []

    test_los = []
    test_acc = []

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            keys = list(logs.keys())
            train_los.append(logs['loss'])
            train_acc.append(logs['sparse_categorical_accuracy'])

        def on_test_batch_end(self, batch, logs=None):
            keys = list(logs.keys())
            test_los.append(logs['loss'])
            test_acc.append(logs['sparse_categorical_accuracy'])
    
    #training
    #define scheduler with warmup
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=ILR,
        decay_steps=D_S,
        decay_rate=D_R)
    #warmup mechanism
    scheduler = WarmUp(initial_learning_rate=ILR,warmup_steps=W_S,decay_schedule_fn=lr_schedule)
    sch_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    #define adamW optimizer
    AdamW = AdamWeightDecay(scheduler)#learning_rate=ILR)

    model.compile(
        optimizer=AdamW,#tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy()
    )
    #callback
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir='callback',update_freq=100,histogram_freq=100),CustomCallback()]#,sch_callback]
    #training
    history = model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=EPOCH, callbacks = callbacks)
    
    #save acc and loss information
    curve = [train_los,train_acc,test_los,test_acc]
    #save model
    #model.save_weights('./Stage_II/weight/'+model_name+dataset_name+'weights')
    return history,curve

#model training pipline
def train_pipline(minor_data,dataset_name,model_name = 'bert-base-uncased'):
    from datasets import Dataset
    #train test dataset split
    dataset_data = Dataset.from_pandas(minor_data)
    dataset_split = dataset_data.train_test_split(test_size=0.2)

    #parameters
    dataset = dataset_split
    MODEL = model_name

    EPOCH       = 6
    ILR         = 5e-5
    D_S         = 600
    D_R         = 0.9
    W_S         = 200

    PARAMETERS = [EPOCH,ILR,D_S,D_R,W_S]

    #################################
    #tokenization                   #
    #################################
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def tokenize_function(examples):
        return tokenizer(examples['review'], padding="max_length", max_length=64, truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    #################################
    #train/test dataset conversion  #
    #################################
    train_dataset = tokenized_datasets["train"].shuffle(seed=1)
    eval_dataset = tokenized_datasets["test"].shuffle(seed=1)

    from transformers import DefaultDataCollator
    data_collator = DefaultDataCollator(return_tensors="tf")

    tf_train_dataset = train_dataset.to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids"],
        label_cols=["rating"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=16,
    )

    tf_validation_dataset = eval_dataset.to_tf_dataset(
        columns=["attention_mask", "input_ids", "token_type_ids"],
        label_cols=["rating"],
        shuffle=False,
        collate_fn=data_collator,
        batch_size=8,
    )
    #################################
    #model init and training        #
    #################################

    #model initialization
    model = model_initialization(MODEL=MODEL)
    #model training
    history,curve = model_train(model,MODEL,tf_train_dataset,tf_validation_dataset,dataset_name,PARAMETERS)
    #check the accuracy curve
    #print(history.params)
    #print(history.history.keys())

    return model,curve,tf_validation_dataset,eval_dataset

#result analysis
def result_analysis(model,tf_validation_dataset,eval_dataset,name):
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    import numpy as np
    path_img = './Stage_II/img'
    #prediction based on trained model
    predict = model.predict(tf_validation_dataset)

    Y_test = eval_dataset['rating']
    y_pred = np.argmax(predict.logits,axis = 1)
    #precision recall f1 socre of each category
    report = classification_report(Y_test, y_pred)
    #open text file
    text_file = open('./Stage_II/report/'+name+'.txt', "w")
    #write string to file
    text_file.write(report)
    #close file
    text_file.close()

    #heatmap
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(Y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig,ax = plt.subplots(figsize = (3.8,2.6))
    plt.rcParams['font.size'] = 9.7

    disp.plot(ax=ax)
    plt.savefig(path_img+'/'+name+'_conmat'+'.png')
    plt.savefig(path_img+'/'+name+'_conmat'+'.svg')
    pass

#draw curve
def learn_curve(data,name):
    import matplotlib.pyplot as plt

    path_img = './Stage_II/img'
    fig,ax1 = plt.subplots(figsize = (3.8,2.6))
    plt.rcParams['font.size'] = 9.7

    loss1 = data[0]
    acc1  = data[1]
    loss2 = data[2]
    acc2  = data[3]

    ax1.set_xlabel("Learning Curve")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Acc")

    ax1.plot(loss1, linewidth = 1.6)
    ax1.plot(loss2, linewidth = 1.6)
    ax2.plot(acc1 , linewidth = 1.6)
    ax2.plot(acc2 , linewidth = 1.6)
    plt.savefig(path_img+'/'+name+'_lecur'+'.png')
    plt.savefig(path_img+'/'+name+'_lecur'+'.svg')
    pass

def model_dataset_choose(code,minor_data_ori,minor_data_bal):
    if   code == 0:
        #original ratio dataset using BERT
        mod_ori_bert,cur_ori_bert,val__ori_bert,eva_ori_bert = train_pipline(minor_data_ori,'ori','bert-base-uncased')
        result_analysis(mod_ori_bert,val__ori_bert,eva_ori_bert,'bert_ori')
        learn_curve(cur_ori_bert,'ori'+'bert-base-uncased')
    elif code == 1:
        #balance ratio dataset using BERT
        mod_bal_bert,cur_bal_bert,val__bal_bert,eva_bal_bert = train_pipline(minor_data_bal,'bal','bert-base-uncased')
        result_analysis(mod_bal_bert,val__bal_bert,eva_bal_bert,'bert_bal')
        learn_curve(cur_bal_bert,'bal'+'bert-base-uncased')
    elif code == 2:
        #original ratio dataset using RoBERTa
        mod_ori_roberta,cur_ori_roberta,val__ori_roberta,eva_ori_roberta = train_pipline(minor_data_ori,'ori','roberta-base')
        result_analysis(mod_ori_roberta,val__ori_roberta,eva_ori_roberta,'roberta_ori')
        learn_curve(cur_ori_roberta,'ori'+'roberta-base')
    elif code == 3:
        #balance ratio dataset using RoBERTa
        mod_bal_roberta,cur_bal_roberta,val__bal_roberta,eva_bal_roberta = train_pipline(minor_data_bal,'bal','roberta-base')
        result_analysis(mod_bal_roberta,val__bal_roberta,eva_bal_roberta,'roberta_bal')
        learn_curve(cur_bal_roberta,'bal'+'roberta-base')
    pass