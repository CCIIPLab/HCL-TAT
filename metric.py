# -*- coding: utf-8 -*-

from sklearn.metrics import precision_score, recall_score, f1_score

class Metric:
    def __init__(self):        
        self.pred = []
        self.true = []

    def update_state(self, preds, trues, id2labels):
        batch_size = len(id2labels)
        
        _, seq_len = trues.shape
        preds = preds.view(batch_size, -1, seq_len)
        trues = trues.view(batch_size, -1, seq_len)

        preds = preds.cpu().tolist()
        trues = trues.cpu().tolist()
        
        for pred, true, id2label in zip(preds, trues, id2labels):
            pred = self.decode(pred, id2label)
            true = self.decode(true, id2label)
            
            self.pred.extend(pred)
            self.true.extend(true) 
    
    def result(self, use_BIO=True):
        if use_BIO:
            return self.score(self.pred, self.true)
        else:
            return self.score_without_BIO(self.pred, self.true)
    
    def reset(self):
        self.pred = []
        self.true = []

    def decode(self, ids, id2label):
        labels = []
        for ins in ids:
            ins_labels = list(map(lambda x: id2label[x], ins))
            labels.append(ins_labels)
        return labels
    
    def score(self, pred_tags, true_tags):
        true_spans = set(self.get_span(true_tags))
        pred_spans = set(self.get_span(pred_tags))
    
        pred_correct = len(true_spans & pred_spans)
        pred_all = len(pred_spans)
        true_all = len(true_spans)
        """
        pred_list = [item for sublist in pred_tags for item in sublist]
        true_list = [item for sublist in true_tags for item in sublist]
        assert len(pred_list) == len(true_list)

        pred_correct, pred_all, true_all = 0, 0, 0
        for i in range(len(true_list)):
            if true_list[i] == 'PAD':
                continue
            if true_list[i] != 'O':
                true_all += 1
                if pred_list[i] == true_list[i]:
                    pred_correct += 1
            if pred_list[i] != 'O' and pred_list[i] != "PAD ":
                pred_all += 1
        """
    
        p = pred_correct / pred_all if pred_all > 0 else 0
        r = pred_correct / true_all if true_all > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        return p, r, f1

    def score_without_BIO(self, pred_tags, true_tags):
        pred_list = [item for sublist in pred_tags for item in sublist]
        true_list = [item for sublist in true_tags for item in sublist]
        assert len(pred_list) == len(true_list)

        pred_correct, pred_all, true_all = 0, 0, 0
        ti_pred_correct, ti_pred_all, ti_true_all = 0, 0, 0
        for i in range(len(true_list)):
            if true_list[i] == 'PAD':
                continue
            if true_list[i] != 'O':
                true_all += 1
                ti_true_all += 1
                if pred_list[i] == true_list[i]:
                    pred_correct += 1
                if pred_list[i] != 'O':
                    ti_pred_correct += 1
            if pred_list[i] != 'O' and pred_list[i] != "PAD ":
                pred_all += 1
                ti_pred_all += 1

        p = pred_correct / pred_all if pred_all > 0 else 0
        r = pred_correct / true_all if true_all > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        ti_p = ti_pred_correct / ti_pred_all if ti_pred_all > 0 else 0
        ti_r = ti_pred_correct / ti_true_all if ti_true_all > 0 else 0
        ti_f1 = 2 * ti_p * ti_r / (ti_p + ti_r) if ti_p + ti_r > 0 else 0

        return p, r, f1, ti_p, ti_r, ti_f1

    def get_span(self, seq):
        if any(isinstance(s, list) for s in seq):
            seq = [item for sublist in seq for item in sublist + ['O']]
        
        prev_tag = 'O'
        prev_type = ''
        begin_offset = 0
        chunks = []
        for i, chunk in enumerate(seq + ['O']):
            tag = chunk[0]
            type_ = chunk.split('-')[-1]
    
            if self.end_of_span(prev_tag, tag, prev_type, type_):
                chunks.append((prev_type, begin_offset, i-1))
            if self.start_of_span(prev_tag, tag, prev_type, type_):
                begin_offset = i
            prev_tag = tag
            prev_type = type_
    
        return chunks
    
    def start_of_span(self, prev_tag, tag, prev_type, type_):
        chunk_start = False
    
        if tag == 'B': chunk_start = True
        if tag == 'S': chunk_start = True
    
        if prev_tag == 'E' and tag == 'E': chunk_start = True
        if prev_tag == 'E' and tag == 'I': chunk_start = True
        if prev_tag == 'S' and tag == 'E': chunk_start = True
        if prev_tag == 'S' and tag == 'I': chunk_start = True
        if prev_tag == 'O' and tag == 'E': chunk_start = True
        if prev_tag == 'O' and tag == 'I': chunk_start = True
    
        if tag != 'O' and tag != '.' and prev_type != type_:
            chunk_start = True
    
        return chunk_start
    
    def end_of_span(self, prev_tag, tag, prev_type, type_):
        chunk_end = False
    
        if prev_tag == 'E': chunk_end = True
        if prev_tag == 'S': chunk_end = True
    
        if prev_tag == 'B' and tag == 'B': chunk_end = True
        if prev_tag == 'B' and tag == 'S': chunk_end = True
        if prev_tag == 'B' and tag == 'O': chunk_end = True
        if prev_tag == 'I' and tag == 'B': chunk_end = True
        if prev_tag == 'I' and tag == 'S': chunk_end = True
        if prev_tag == 'I' and tag == 'O': chunk_end = True
    
        if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
            chunk_end = True
    
        return chunk_end
