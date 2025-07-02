import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import datetime
from concurrent.futures import ThreadPoolExecutor
import queue
import os
import csv
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import praw
import requests
import threading

# ========== Reddit API Setup ==========
reddit = praw.Reddit(
    client_id="You dont get to see my info",
    client_secret="You dont get to see my info",
    user_agent="You dont get to see my info",
    username="You dont get to see my info",
    password="You dont get to see my info"
)

# ========== NewsAPI Setup ==========
NEWSAPI_KEY = "You dont get to see my info"

# ========== Device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== RSI and Moving Averages Calculations ==========
def calculate_rsi(series, period=14):
    delta = np.diff(series, axis=0)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = np.convolve(up.flatten(), np.ones(period)/period, mode='valid')
    roll_down = np.convolve(down.flatten(), np.ones(period)/period, mode='valid')
    rs = roll_up / (roll_down + 1e-6)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    pad_len = len(series) - len(rsi)
    rsi_padded = np.concatenate([np.full(pad_len, 50), rsi])  # start RSI near 50
    return rsi_padded.reshape(-1, 1)

def moving_average(series, window):
    ma = np.convolve(series.flatten(), np.ones(window)/window, mode='valid')
    pad_len = len(series) - len(ma)
    ma_padded = np.concatenate([np.full(pad_len, ma[0]), ma])
    return ma_padded.reshape(-1, 1)

# ========== Model ==========
class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=100, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ========== Data Download & Feature Engineering ==========
def download_data(symbol, years=10):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365*years)
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    close = df['Close'].values.reshape(-1, 1)
    volume = df['Volume'].values.reshape(-1, 1)
    rsi = calculate_rsi(close)
    ma7 = moving_average(close, 7)
    ma21 = moving_average(close, 21)
    features = np.hstack([close, volume, rsi, ma7, ma21])
    return features

def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, 0])  # predict Close only
    return np.array(X), np.array(y)

# ========== Save/Load Model ==========
def save_model(model, symbol):
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{symbol}_lstm.pth")

def load_model(model, symbol):
    path = f"models/{symbol}_lstm.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        return True
    return False

# ========== Logger ==========
class GuiLogger:
    def __init__(self, text_widget, root):
        self.text_widget = text_widget
        self.root = root
        self.queue = queue.Queue()
        self.update_interval = 100
        self._poll()
    def log(self, msg):
        self.queue.put(msg)
    def _poll(self):
        while not self.queue.empty():
            msg = self.queue.get_nowait()
            self.text_widget.insert(tk.END, msg)
            self.text_widget.see(tk.END)
        self.root.after(self.update_interval, self._poll)

# ========== Sentiment Analysis ==========
def get_reddit_sentiment(symbol):
    try:
        subreddit = reddit.subreddit('stocks')
        sentiments = []
        for submission in subreddit.search(symbol, limit=15):
            title = submission.title
            pos_words = ['gain', 'bull', 'buy', 'long', 'up', 'green']
            neg_words = ['loss', 'bear', 'sell', 'short', 'down', 'red']
            score = 0
            title_lower = title.lower()
            for w in pos_words:
                if w in title_lower:
                    score += 1
            for w in neg_words:
                if w in title_lower:
                    score -= 1
            sentiments.append(score)
        if len(sentiments) == 0:
            return 0
        avg_sentiment = sum(sentiments) / len(sentiments)
        return avg_sentiment
    except Exception:
        return 0

def get_newsapi_sentiment(symbol):
    try:
        url = ('https://newsapi.org/v2/everything?'
               f'q={symbol}&'
               'language=en&'
               'sortBy=publishedAt&'
               'pageSize=15&'
               f'apiKey={NEWSAPI_KEY}')
        r = requests.get(url)
        data = r.json()
        sentiments = []
        if data.get("articles"):
            for art in data["articles"]:
                title = art.get("title", "")
                pos_words = ['gain', 'bull', 'buy', 'long', 'up', 'green']
                neg_words = ['loss', 'bear', 'sell', 'short', 'down', 'red']
                score = 0
                title_lower = title.lower()
                for w in pos_words:
                    if w in title_lower:
                        score += 1
                for w in neg_words:
                    if w in title_lower:
                        score -= 1
                sentiments.append(score)
        if len(sentiments) == 0:
            return 0
        avg_sentiment = sum(sentiments) / len(sentiments)
        return avg_sentiment
    except Exception:
        return 0

# ========== Global Results ==========
stored_results = {}
result_queue = queue.Queue()

# ========== Training & Prediction ==========
def train_and_predict(symbol, days, logger, use_loaded_model=False, preloaded_model=None):
    try:
        logger.log(f"\n▶ Downloading {symbol} data...\n")
        features = download_data(symbol)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(features)

        window_size = 60
        X, y = create_sequences(data_scaled, window_size)
        if len(X) == 0:
            raise ValueError("Not enough data after windowing")

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

        split = int(len(X_tensor) * 0.8)
        X_train, y_train = X_tensor[:split], y_tensor[:split]
        X_val, y_val = X_tensor[split:], y_tensor[split:]

        if preloaded_model:
            model = preloaded_model.to(device)
            logger.log(f"{symbol} ✅ Using loaded model.\n")
        else:
            model = StockLSTM(input_size=features.shape[1]).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if not use_loaded_model:
            if not load_model(model, symbol):
                for epoch in range(1, 101):
                    model.train()
                    optimizer.zero_grad()
                    out = model(X_train)
                    loss = criterion(out, y_train)
                    loss.backward()
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_out = model(X_val)
                        val_loss = criterion(val_out, y_val)

                    if epoch == 1 or epoch % 10 == 0:
                        logger.log(f"{symbol} Epoch {epoch:3}/100 - Loss: {loss.item():.6f} | Val: {val_loss.item():.6f}\n")

                save_model(model, symbol)
                logger.log(f"{symbol} ✅ Model trained & saved.\n")
            else:
                logger.log(f"{symbol} ✅ Loaded existing model.\n")

        # Prepare input sequence for prediction
        input_seq = torch.tensor(data_scaled[-window_size:], dtype=torch.float32).unsqueeze(0).to(device)

        preds = []
        model.eval()
        for _ in range(days):
            with torch.no_grad():
                pred = model(input_seq)
            preds.append(pred.item())
            # Prepare next input sequence by removing first and appending prediction with other features same as last day
            last_features = input_seq[:, -1, :].clone()
            last_features[0, 0] = pred.item()  # replace Close with prediction
            next_input = torch.cat((input_seq[:, 1:, :], last_features.unsqueeze(1)), dim=1)
            input_seq = next_input

        # Inverse transform predictions
        dummy = np.zeros((days, features.shape[1]))
        dummy[:, 0] = np.array(preds)
        preds_inv = scaler.inverse_transform(dummy)[:, 0]

        # Confidence: stddev of predictions (simple proxy)
        confidence = np.std(preds_inv)

        # Fetch sentiment asynchronously
        sentiment_data = {}
        def fetch_sentiment():
            sentiment_data['reddit'] = get_reddit_sentiment(symbol)
            sentiment_data['newsapi'] = get_newsapi_sentiment(symbol)
        t = threading.Thread(target=fetch_sentiment)
        t.start()
        t.join(timeout=10)

        reddit_sent = sentiment_data.get('reddit', 0)
        news_sent = sentiment_data.get('newsapi', 0)
        avg_sentiment = (reddit_sent + news_sent) / 2

        # Queue results for GUI update
        result_queue.put((symbol, days, preds_inv, confidence, avg_sentiment))
        logger.log(f"{symbol} ✅ Done.\n")

    except Exception as e:
        logger.log(f"\n{symbol} ERROR: {e}\n")
        logger.log(f"{symbol} ✅ Done.\n")

# ========== GUI Functions ==========
def start_predictions():
    symbols_raw = symbol_entry.get().upper().replace(' ', '')
    symbols = [s.strip() for s in symbols_raw.split(',') if s.strip()]
    try:
        days = int(days_entry.get())
        if not (1 <= days <= 365):
            raise ValueError
    except ValueError:
        messagebox.showerror("Invalid Input", "Days must be an integer between 1 and 365.")
        return

    output_text.delete(1.0, tk.END)
    clear_plot()

    max_workers = 5
    executor = ThreadPoolExecutor(max_workers=max_workers)
    for sym in symbols:
        executor.submit(train_and_predict, sym, days, gui_logger)

def load_model_for_stock():
    symbol = symbol_entry.get().strip().upper()
    if not symbol:
        messagebox.showerror("Input Error", "Please enter a stock symbol to load the model.")
        return
    model = StockLSTM(input_size=5).to(device)
    if load_model(model, symbol):
        messagebox.showinfo("Model Loaded", f"Model for {symbol} loaded successfully.")
        days = simpledialog.askinteger("Prediction Days", "Enter days to predict (1-365):", minvalue=1, maxvalue=365)
        if days:
            output_text.delete(1.0, tk.END)
            clear_plot()
            executor = ThreadPoolExecutor(max_workers=1)
            executor.submit(train_and_predict, symbol, days, gui_logger, True, model)
    else:
        messagebox.showwarning("Load Failed", f"No saved model found for {symbol}.")

def clear_plot():
    global canvas, figure, ax
    if canvas:
        canvas.get_tk_widget().destroy()
        canvas = None
        figure = None
        ax = None
    stored_results.clear()
    summary_label.config(text="")

def plot_results():
    global canvas, figure, ax
    if canvas is None:
        figure = Figure(figsize=(8,4))
        ax = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=plot_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    while not result_queue.empty():
        try:
            symbol, days, preds, confidence, sentiment = result_queue.get_nowait()
            stored_results[symbol] = (days, preds, confidence, sentiment)
        except queue.Empty:
            break

    ax.clear()
    for symbol, (days, preds, confidence, sentiment) in stored_results.items():
        ax.plot(range(1, days+1), preds, label=symbol)

    ax.set_title("Predicted Prices")
    ax.set_xlabel("Days in Future")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    canvas.draw()

    # Show summary with confidence & sentiment
    summary_lines = []
    for symbol, (days, preds, confidence, sentiment) in stored_results.items():
        if len(preds) >= 2:
            start_price = preds[0]
            end_price = preds[-1]
            pct_change = ((end_price - start_price) / start_price) * 100
            direction = "up" if pct_change > 0 else "down"
            sentiment_str = f"{sentiment:.2f}"
            summary_lines.append(
                f"{symbol}: AI predicts {direction} {abs(pct_change):.2f}% in {days} days. "
                f"Confidence (stddev): {confidence:.2f}. Sentiment: {sentiment_str} (pos > 0)."
            )
        else:
            summary_lines.append(f"{symbol}: Not enough prediction data.")

    summary_label.config(text="\n".join(summary_lines))

    root.after(500, plot_results)

# ========== Autocomplete Entry ==========
class AutocompleteEntry(tk.Entry):
    def __init__(self, ticker_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ticker_list = sorted(ticker_list)
        self.var = self["textvariable"] = tk.StringVar()
        self.var.trace('w', self._on_change)
        self.lb = None

    def _on_change(self, *args):
        val = self.var.get()
        if ',' in val:
            last = val.split(',')[-1].strip()
        else:
            last = val.strip()
        if last == '':
            self._hide_listbox()
            return
        matches = [t for t in self.ticker_list if t.startswith(last.upper())]
        if matches:
            if not self.lb:
                self.lb = tk.Listbox()
                self.lb.bind("<<ListboxSelect>>", self._on_select)
                self.lb.place(x=self.winfo_x(), y=self.winfo_y()+self.winfo_height())
            self.lb.delete(0, tk.END)
            for m in matches[:10]:
                self.lb.insert(tk.END, m)
        else:
            self._hide_listbox()

    def _on_select(self, event):
        if not self.lb:
            return
        sel = self.lb.get(tk.ACTIVE)
        val = self.var.get()
        if ',' in val:
            parts = val.split(',')
            parts[-1] = sel
            new_val = ','.join(parts)
        else:
            new_val = sel
        self.var.set(new_val)
        self._hide_listbox()
        self.icursor(tk.END)

    def _hide_listbox(self):
        if self.lb:
            self.lb.destroy()
            self.lb = None

# ========== Load tickers from CSV ==========
def load_tickers(filename="tickers.csv"):
    tickers = []
    if os.path.exists(filename):
        with open(filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sym = row.get('Symbol') or row.get('symbol') or row.get('Ticker')
                if sym:
                    tickers.append(sym)
    else:
        messagebox.showwarning("Ticker file missing", f"Ticker file {filename} not found. Autocomplete disabled.")
    return tickers

# ========== GUI Setup ==========
root = tk.Tk()
root.title("📈 LSTM Stock Predictor with RSI, MA, Volume, Sentiment & Confidence")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10, fill=tk.X)

tk.Label(frame, text="Enter stock symbols (comma separated):").pack(anchor='w')
tickers = load_tickers()
symbol_entry = AutocompleteEntry(tickers, frame, width=60)
symbol_entry.pack(pady=5, fill=tk.X)
symbol_entry.insert(0, "AAPL,TSLA")

tk.Label(frame, text="Days to predict (1-365):").pack(anchor='w')
days_entry = tk.Entry(frame, width=10)
days_entry.pack(pady=5)
days_entry.insert(0, "30")

button_frame = tk.Frame(frame)
button_frame.pack(pady=10, fill=tk.X)

predict_button = tk.Button(button_frame, text="Start Prediction", command=start_predictions)
predict_button.pack(side=tk.LEFT, padx=(0,10))

load_model_button = tk.Button(button_frame, text="Load Model & Predict", command=load_model_for_stock)
load_model_button.pack(side=tk.LEFT)

output_text = scrolledtext.ScrolledText(frame, height=15, width=80)
output_text.pack(fill=tk.BOTH, expand=True)

gui_logger = GuiLogger(output_text, root)

summary_label = tk.Label(root, text="", justify="left", font=("Arial", 10))
summary_label.pack(padx=10, pady=(10,5), anchor='w')

plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

canvas = None
figure = None
ax = None

root.after(500, plot_results)

root.mainloop()
