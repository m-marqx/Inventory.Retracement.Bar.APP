import tkinter as tk


class Text:
    def __init__(self, master):
        self.text_widget = tk.Text(master, height=8, width=80)
        self.text_widget.grid(row=2, column=0, columnspan=4)
        self.text_widget.configure(state="disabled")

    def insert_text(self, text):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, f"{text}\n")
        self.text_widget.configure(state="disabled")
        return self
