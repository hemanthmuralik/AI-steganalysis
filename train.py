import argparse, os, numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

# TLU activation
def tlu(x, threshold=3.0):
    return tf.maximum(tf.minimum(x, threshold), -threshold)

# Custom callback to compute Detection Error P_E on validation set each epoch
class ValidationPECallback(Callback):
    def __init__(self, val_gen, out_path='checkpoints', monitor='val_pe'):
        super().__init__()
        self.val_gen = val_gen
        self.best_pe = 1.0
        self.out_path = out_path

    def on_epoch_end(self, epoch, logs=None):
        # gather predictions and labels
        y_true = []
        y_pred = []
        for X, y in self.val_gen:
            preds = self.model.predict(X, verbose=0)
            y_true.append(y)
            y_pred.append(preds.reshape(-1))
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        # threshold at 0.5
        y_hat = (y_pred >= 0.5).astype('int32')
        tp = ((y_hat==1) & (y_true==1)).sum()
        tn = ((y_hat==0) & (y_true==0)).sum()
        fp = ((y_hat==1) & (y_true==0)).sum()
        fn = ((y_hat==0) & (y_true==1)).sum()
        # rates (avoid div by zero)
        n_pos = max(1, int((y_true==1).sum()))
        n_neg = max(1, int((y_true==0).sum()))
        false_positive_rate = fp / n_neg
        false_negative_rate = fn / n_pos
        pe = 0.5 * (false_positive_rate + false_negative_rate)
        logs = logs or {}
        logs['val_pe'] = pe
        print(f"\nEpoch {epoch+1} Validation P_E = {pe:.4f} (FPR={false_positive_rate:.4f}, FNR={false_negative_rate:.4f})")
        # save best by val_pe (lower is better)
        if pe < self.best_pe:
            self.best_pe = pe
            os.makedirs(self.out_path, exist_ok=True)
            fname = os.path.join(self.out_path, 'best_pe.h5')
            self.model.save(fname, include_optimizer=False)
            print('Saved best model to', fname)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--covers', required=True)
    parser.add_argument('--stegos', required=True)
    parser.add_argument('--model', choices=['ye','srnet'], default='ye')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--out', default='checkpoints')
    args = parser.parse_args()

    from data.generator import list_images, StegoGenerator
    cover_paths = list_images(args.covers)
    stego_paths = list_images(args.stegos)
    assert len(cover_paths) == len(stego_paths), 'Covers and stegos must be same length and paired.'

    # split by cover (prevent leakage)
    n = len(cover_paths)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n*0.8)
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_covers = [cover_paths[i] for i in train_idx]
    train_stegos = [stego_paths[i] for i in train_idx]
    val_covers = [cover_paths[i] for i in val_idx]
    val_stegos = [stego_paths[i] for i in val_idx]

    train_gen = StegoGenerator(train_covers, train_stegos, batch_size=args.batch, augment=True, gray=True)
    val_gen = StegoGenerator(val_covers, val_stegos, batch_size=args.batch, augment=False, gray=True)

    if args.model == 'ye':
        from models.ye_net import ye_net
        model = ye_net(input_shape=(256,256,1))
    else:
        from models.srnet_like import srnet_like
        model = srnet_like(input_shape=(256,256,1))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    model.summary()

    # callbacks
    pe_cb = ValidationPECallback(val_gen, out_path=args.out)
    es = EarlyStopping(monitor='val_pe', mode='min', patience=8, verbose=1)
    # note: ModelCheckpoint can't directly monitor val_pe; we save in callback based on pe.
    model.fit(train_gen, epochs=args.epochs, callbacks=[pe_cb, es], verbose=1)

if __name__ == '__main__':
    main()
