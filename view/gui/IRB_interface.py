from view.gui import Label, Grid, StrategyButton, GetDataGUI


class Interface:
    def __init__(self, master):
        self.master = master
        self.master.title("Futures Trading Strategy")

        self.label = Label(self.master)
        self.label_grid = Grid(self.master)

        self.strategy_run = GetDataGUI(self.master)
        self.strategy_button = StrategyButton(self.master)
