""" Summary
"""

import os
import time
import numpy as np
import datetime
from contextlib import ContextDecorator

class StylerBox:
    def __init__(self, string=None):
        self.string = string
        return

    def __unicodify(self, ucd):
        return f"{ucd}{self.string}\033[0m"

    def boldface(self, string=None):
        self.string = self.__unicodify("\033[1m")
        return self

    def red(self, string=None):
        self.string = self.__unicodify("\033[91m")
        return self

    def yellow(self, string=None):
        self.string = self.__unicodify("\033[92m")
        return self


class Timer(ContextDecorator):
    def __init__(self):
        return

    def __enter__(self):
        self.begin_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.time = time.time() - self.begin_time


class ProgressBar:
    """_summary_
    """

    def __init__(self, num_iters, header=None, bar_prefix="") -> None:
        self._t0 = time.time()
        self._n = num_iters
        self.term_cols = (os.get_terminal_size().columns)*0.15
        self.bar_prefix = bar_prefix

        if header is not None:
            print(header)
        print(f"[{'.'*int(self.term_cols/self._n)}] loop started..!", end="\r")

    def update(self, i: int, bar_prefix: str = None) -> None:
        """_summary_

        :param i: _description_
        :type i: int
        """
        if bar_prefix is None:
            bar_prefix = self.bar_prefix
        tc_points = int(((i+1)/self._n) * self.term_cols)
        avg_time = (time.time()-self._t0)/(i+1)
        percent = ((i+1)/self._n)*100
        #
        completion = f"{percent :04.2f}%"
        remaining_time = f" | TTF: {avg_time*(self._n-(i+1)):04.1f}s"
        bar_str = f"{bar_prefix}[{'='*(tc_points)+'>'+'.'*int(self.term_cols-tc_points)}]"
        print(f"{bar_str}{completion} {remaining_time}", end="\r")


def nparray_summary(arrays: dict[str, np.ndarray], print_summary=True):
    _summary = [""]
    for (k, v) in arrays.items():
        _summary.append(f"{k}:")
        _summary.append(f"\tShape {v.shape}")
        _summary.append(f"\tData type {v.dtype}")
        _summary.append(f"\tMax {np.min(v)}")
        _summary.append(f"\tMax {np.max(v)}")
    if print_summary:
        print("\n".join(_summary))
    else:
        return _summary


def error_msg(msg: str):
    return f"\033[1m\033[91m{msg}\033[0m"


class SimpleTable:
    """_summary_
    """

    def __init__(
        self,
        vertical_marker: str = "|",
        horizontal_marker: str = "_",
        alignment: str = "^",
        line_len: int = 80,
        floats: int = 3,
        title: str = None,
    ):
        self.vmarker = vertical_marker
        self.hmarker = horizontal_marker
        self.line_len = line_len
        self.col_widths = []
        self.floats = floats
        self.alignment: str = alignment
        self.title = title
        #
        self.rows = []
        #
        self.nrows = None
        self.ncols = None

    _superscore = u'\u203e'
    
    def _make_hline(self, top=True, bot=True):
        if self.col_widths is None:
            raise ValueError(error_msg("col_wdiths are not available!"))
        _hl_symb = "+"
        _hl_thread = "-"
        if top:
            self._hline_top = f"{_hl_symb}" + f"{_hl_symb}".join(
                [f"{_hl_thread*i}" for i in self.col_widths]
            ) + f"{_hl_symb}"
        if bot:
            self._hline_bottom = f"{_hl_symb}" + f"{_hl_symb}".join(
                [f"{_hl_thread*i}" for i in self.col_widths]
            ) + f"{_hl_symb}"
        return self

    def add_column(self):
        """_summary_
        """
        raise NotImplementedError(
            error_msg("Adding column wise is not available!"))

    def _format_fields(self):
        for (_i, _acol) in enumerate(self.content):
            for (_k, _afield) in enumerate(_acol):
                if type(_afield) == float:
                    _acol[_k] = f"{_afield:0.{self.floats}f}"
                elif type(_afield) in (int, str):
                    _acol[_k] = f"{_afield}"
                else:
                    raise TypeError(
                        error_msg(f"{type(_afield)} type fields are not allowed!"))
            self.col_widths.append(max([len(i) for i in _acol]) + 4)
            for (_j, _afield) in enumerate(_acol):
                fmt_prefix = f"{self.alignment}{self.col_widths[-1]}"
                self.content[_i][_j] = f"{_afield:{fmt_prefix}}"
        return self

    def _make_content_column_type(self,):
        _col_type_content = []
        for a_col in zip(*self.content):
            _col_type_content.append(list(a_col))
        self.content = _col_type_content
        return self

    _acceptable_content_type = ("COLUMN", "ROW")

    def __call__(
        self,
        content: list[list] = None,
        content_type="COLUMN",
        make_header: bool = True,
        make_footer: bool = True,
        foot_notes: dict = None,
    ):
        self.content = content
        assert content_type in self._acceptable_content_type, error_msg(
            f"Content type must be a  kind of {self._acceptable_content_type}"
            f"but {content_type} found."
        )
        #
        if content_type == "ROW":
            self._make_content_column_type()
        self.ncols = len(self.content)
        self.nrows = len(self.content[0])
        #
        self._format_fields()
        self._make_hline()
        #
        _vmrkr = f"{self.vmarker}"
        for arow_elements in zip(*self.content):
            self.rows.append(f"{_vmrkr}" + f"{_vmrkr}".join(arow_elements)+f"{_vmrkr}")
        #
        self.row_len = sum([len(i) for i in self.rows[0]])
        if make_header:
            self.rows.insert(0, self._hline_top)
            self.rows.insert(2, self._hline_bottom)
        if make_footer:
            self.rows.append(self._hline_bottom)
        #
        if foot_notes:
            self.rows.append("NOTE:")
            for (k, v) in foot_notes.items():
                self.rows.append(f"\t{k}: {v}")
        #
        if self.title is not None:
            self.rows.insert(0, self.title)
        #
        self.table = "\n".join(self.rows)
        #
        return self.table


class Report:
    """_summary_
    """

    def __init__(self, file_path: str = None, mow: str = "w", line_len: int = 90, line_marker="=", subhead_marker="-", report_title='REPORT'):
        self.report: list = []
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

        now = datetime.datetime.now()
        now_time = f"Time: {now.strftime('%H:%M:%S')}"
        now_date = f"Date: {now.strftime('%d-%m-%Y')}"
        self.report.append(
            f"{now_time}{' '*(self.line_len-len(now_time+now_date))}{now_date}\n\n"
        )

    def sub_heading(self, sh: str):
        return f"\n\n{sh}\n{self.sh_marker*len(sh)}"


class TrainingReport(Report):
    """_summary_
    """

    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.init(title="MODEL TRAINING REPORT")
        return

    def add_data_set_summary(self, nn_model):
        self.report += ["", nn_model.dataset_summary, ""]
        return self

    def add_hyperparameters(self):
        if self.model is None:
            raise ValueError(
                error_msg("Model is missing for writing hyper parameters."))
        self.report.append(self.sub_heading(f"PARAMETERS:"))
        self.report.append(f"\tNumber of epochs: {self.model.num_epochs}")
        self.report.append(f"\tBatch Size: {self.model.batch_size}")
        self.report.append(f"\tOptimizer: {self.model.optimizer._name}")
        self.report.append(
            f"\tOptimizer Learning Rate: {self.model.optimizer.learning_rate.numpy():8.6f}")
        self.report.append(f"\tLoss Function: {self.model.loss.name}")
        return self

    def add_model_summary(self, nn_model):
        self.report.append(self.sub_heading(f"MODEL SUMMARY:"))
        nn_model.summary(print_fn=lambda x: self.report.append(
            "\t"+x), line_length=self.line_len)
        return self

    def add_model_performance(self, nn_model):
        self.report.append(self.sub_heading(f"TRAINING HISTORY:"))
        metrics = [_ametric.name for _ametric in nn_model.metrics]
        #
        performance_info = [
            ["Epochs", ] + nn_model.training_report["epochs"],
            ["Iterations", ] + nn_model.training_report["iters"],
            ["TR-loss"] + nn_model.training_report["loss"],
            ["VAL-loss"] + nn_model.validation_report["loss"],
        ]
        #
        for (_i, a_metric) in enumerate(metrics):
            performance_info.append(
                [f"tr-metric-{_i+1}", ] + nn_model.training_report[a_metric]
            )
        #
        for (_j, a_metric) in enumerate(metrics):
            performance_info.append(
                [f"val-metric-{_j+1}", ] + nn_model.validation_report[a_metric]
            )
        metrics_foot_notes = {
            f"metric-{i+1}": _ametric for (i, _ametric) in enumerate(metrics)
        }
        #
        self.report.append(
            SimpleTable(floats=4)(
                content=performance_info,
                content_type="COLUMN",
                make_footer=True,
                make_header=True,
                foot_notes=metrics_foot_notes,
            )
        )
        #
        self.report.append(self.sub_heading(f"TEST PERFORMANCE:"))
        test_performance = [
            ["loss"] + nn_model.testing_report["loss"],
        ]
        for a_metric in nn_model.metrics:
            test_performance.append(
                [f"metric-{_i+1}", ] + nn_model.testing_report[a_metric.name]
            )
        self.report.append(SimpleTable()(test_performance))
        #
        return self

    def write_to_file(self, fpath=None, mow=None):
        if not fpath:
            fpath = self.fpath
        if not mow:
            mow = self.mow
        if fpath is not None:
            with open(fpath, mow) as fh:
                fh.write("\n".join(self.report))
