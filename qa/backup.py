
class XLMRobertaConfig(RobertaConfig):
    pretrained_config_archive_map = XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP


class XLMRobertaModel(RobertaModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            eo match pre-training, XLM-RoBERTa input sequence should be formatted with <s> and </s> tokens as follows:

            (a) For sequence pairs:

                ``tokens:         <s> is this jack ##son ##ville ? </s> </s> no it is not . </s>``

                ``token_type_ids:   0   0  0    0    0     0       0   0   0     1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         <s> the dog is hairy . </s>``

                ``token_type_ids:   0   0   0   0  0     0   0``

            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaModel.from_pretrained('xlm-roberta-large')
        input_ids = torch.tensor(tokenizer.encode("Schloß Nymphenburg ist sehr schön .")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class XLMRobertaForMaskedLM(RobertaForMaskedLM):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-large')
        input_ids = torch.tensor(tokenizer.encode("Schloß Nymphenburg ist sehr schön .")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-large')
        input_ids = torch.tensor(tokenizer.encode("Schloß Nymphenburg ist sehr schön .")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class XLMRobertaForMultipleChoice(RobertaForMultipleChoice):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaForMultipleChoice.from_pretrained('xlm-roberta-large')
        choices = ["Schloß Nymphenburg ist sehr schön .", "Der Schloßkanal auch !"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class XLMRobertaForTokenClassification(RobertaForTokenClassification):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-large')
        input_ids = torch.tensor(tokenizer.encode("Schloß Nymphenburg ist sehr schön .", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


class XLMRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **is_impossible**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels whether a question has an answer or no answer (SQuAD 2.0)
        **cls_index**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the classification token to use as input for computing plausibility of the answer.
        **p_mask**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...)
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
        model = XLMForQuestionAnswering.from_pretrained('xlm-mlm-en-2048')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss, start_scores, end_scores = outputs[:2]
    """
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP


def find_similarities_query_spt_old(tokenizer, model_trans, spt_examples, qry_examples):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    model = SentenceTransformer("xlm-r-distilroberta-base-paraphrase-v1")

    spt_lm_outputs = []
    for spt in tqdm(spt_examples[:100]):
        spt_tokens = ['[CLS]'] + tokenizer.tokenize(spt.question_text) \
                     + ['[SEP]'] + tokenizer.tokenize(spt.context_text) \
                     + ['[SEP]'] + tokenizer.tokenize(spt.answer_text) + ['[SEP]']

        spt_token_ids = tokenizer.convert_tokens_to_ids(spt_tokens)
        if len(spt_token_ids) > 512:
            spt_token_ids = spt_token_ids[:512]

        spt_token_tensor = LongTensor([spt_token_ids])
        #print("spt_token_tensor.shape:", spt_token_tensor.shape)

        model_trans.eval()
        with torch.no_grad():
            lm_output = model_trans(spt_token_tensor)[0][:, 0, :]#[:, 0, :]

        #print("lm_output.shape:", lm_output.shape)
        spt_lm_outputs.append(lm_output)

    for qry in qry_examples:
        qry_tokens = ['[CLS]'] + tokenizer.tokenize(qry.question_text) \
                     + ['[SEP]'] + tokenizer.tokenize(qry.context_text) \
                     + ['[SEP]'] + tokenizer.tokenize(qry.answer_text) + ['[SEP]']

        qry_token_ids = tokenizer.convert_tokens_to_ids(qry_tokens)

        if len(qry_token_ids) > 512:
            qry_token_ids = qry_token_ids[:512]

        qry_token_tensor = LongTensor([qry_token_ids])

        model_trans.eval()
        with torch.no_grad():
            lm_output = model_trans(qry_token_tensor)[0]

            #euclidean_dist = torch.stack(tuple([lm_output_qry.sub(lm_output_spt).pow(2).sum(dim=-1)
            #                                   for lm_output_spt in spt_lm_outputs]))

            lm_output_qry = lm_output[:, 0, :]

            cos_dis = torch.stack([cos(lm_output_qry, lm_output_spt) for lm_output_spt in spt_lm_outputs]).flatten()

            rankings = torch.argsort(cos_dis, dim=0, descending=True).tolist()

        qry.set_rankings(rankings[:10])


print("TRAINING FROM SCRATCH ...")

global_step, epochs_trained, tr_loss, logging_loss = 1, 0, 0.0,  0.0

model.zero_grad()

train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch", disable=local_rank not in [-1, 0])

for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        model.train()
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": None if model_type in ["xlm", "xlm-roberta", "distilbert"] else batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        if model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
            if version_2_with_negative:
                inputs.update({"is_impossible": batch[7]})
        if model_type == "xlm":
            inputs["langs"] = batch[7]
        outputs = model(**inputs)
        # model outputs are always tuple in transformers (see doc)
        loss = outputs[0]

        loss = loss.mean()

        if pre_train_config["gradient_accumulation_steps"] > 1:
            loss = loss / pre_train_config["gradient_accumulation_steps"]

        loss.backward()

        tr_loss += loss.item()

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()

        if (step + 1) % pre_train_config["gradient_accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), pre_train_config["max_grad_norm"])

            global_step += 1

            ## Write loss metrics
            writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
            writer.add_scalar("TRAIN_loss", (tr_loss - logging_loss) / pre_train_config["logging_steps"],
                              global_step)
            logging_loss = tr_loss

            ## Evaluation on train, test datasets
            #train_results = evaluate(tokenizer, model, train_features, train_examples, train_dataset,
            #                         ",".join(train_langs), "train", pre_train_config["eval_batch_size"],
            #                         model_type, out_dir, pre_train_config["n_best_size"],
            #                         pre_train_config["max_answer_length"], version_2_with_negative,
            #                         verbose_logging, do_lower_case, null_score_diff_threshold, lang2id)

            #print("train_results:", train_results)
            #for key, value in train_results.items():
            #    writer.add_scalar("train_{}".format(key), value, global_step)

        if pre_train_config["save_steps"] > 0 and global_step % pre_train_config["save_steps"] == 0:
            output_dir = os.path.join(out_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)


print("MULTILINGUAL TESTING ...")
for lang in test_langs:
    test_results = evaluate(tokenizer, model, test_features[lang], test_examples[lang], test_dataset[lang],
                            lang, "test", pre_train_config["eval_batch_size"],
                            model_type, out_dir, pre_train_config["n_best_size"],
                            pre_train_config["max_answer_length"], version_2_with_negative,
                            verbose_logging, do_lower_case, null_score_diff_threshold, lang2id)

    print("lang:", lang, " test_results:", test_results)
    for key, value in test_results.items():
        writer.add_scalar("Test_{}_{}".format(lang, key), value, 0)

####


