"""_summary_

:return: _description_
:rtype: _type_
"""
import datetime

class Report:
    """_summary_
    """

    def __init__(self, file_path: str=None, mow: str = "w", line_len: int = 90, line_marker="=", subhead_marker="-", report_title='REPORT'):
        self.report = []
        self.fpath = file_path
        self.mow = mow
        self.line_len = line_len
        self.line_marker = line_marker
        self.sh_marker = subhead_marker
        self.report_title = report_title

    def hline(self, marker=None, line_len=None):
        if not marker:
            marker = self.line_marker
        if not line_len:
            line_len = self.line_len
        return f"{marker*line_len}"

    def init(self, title='Report'):

        self.report.append(self.hline())
        self.report.append(f"{self.report_title:^{self.line_len}}")
        self.report.append(self.hline())

        now=datetime.datetime.now()
        now_time = f"Time: {now.strftime('%H:%M:%S')}"
        now_date = f"Date: {now.strftime('%d-%m-%Y')}"
        self.report.append(
            f"{now_time}{' '*(self.line_len-len(now_time+now_date))}{now_date}"
        )


    def sub_heading(self, sh: str):
        return f"\n{sh}\n{self.sh_marker*len(sh)}"


class TrainingReport(Report):
    """_summary_
    """

    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.init(title="MODEL TRAINING REPORT")
        return

    def add_hyperparameters(self):
        if self.model is None:
            raise ValueError("Model is missing for writing hyper parameters.")       
        self.report.append(self.sub_heading(f"HYPERPARAMETERS:"))
        self.report.append(f"\tNumber of epochs: {self.model.num_epochs}")
        self.report.append(f"\tBatch Size: {self.model.batch_size}")
        self.report.append(f"\tOptimizer: {self.model.optimizer._name}")
        self.report.append(f"\tOptimizer Learning Rate: {self.model.optimizer.learning_rate.numpy():8.6f}")
        self.report.append(f"\tLoss Function: {self.model.loss.name}")
        return self

    def add_model_summary(self, nn_model):
        self.report.append(self.sub_heading(f"MODEL SUMMARY:"))
        nn_model.summary(print_fn=lambda x: self.report.append("\t"+x), line_length=self.line_len)
        return self

    def add_model_performance(self, nn_model):
        self.report.append(self.sub_heading(f"TRAINING HISTORY:"))
        self.report.append(f"\tTraining time: {self.model.training_time: 5.3f} seconds")
        self.report += [f"\t{i}" for i in nn_model.training_history]
        return self

    def write_to_file(self, fpath=None, mow=None):
        if not fpath:
            fpath = self.fpath
        if not mow:
            mow = self.mow
        if fpath is not None:
            with open(fpath, mow) as fh:
                fh.write("\n".join(self.report))

class NeatTable:
    """_summary_
    """

    def __init__(
        self,
        sep: str = "|",
        head_symbol: str = "*",
        titles: list[str] = None,
    ):
        self.sep = sep
        self.head_symbol = head_symbol
        self.titles = titles
        self.num_columns = len(titles)

    def add_column(self):
        """_summary_
        """

    def add_row(self):
        """_summary_

        :param vals: _description_
        :type vals: np.ndarray
        """

    def make_header(self, titles: list[str] = None):
        if self.titles is None:
            assert titles is not None, "Please provide titles!"
            self.titles = titles
            self.num_columns = len(titles)
        header = ""
        for atitle in self.titles:
            header += f"{self.sep} {atitle} {self.sep}"
        hline = self.head_symbol*len(header)
        header = f"{hline}\n{header}\n{hline}\n"
        return header

