import os
from NeuralNetworks.report import TrainingReport


CFD = os.path.dirname(__file__)
report = TrainingReport(file_path=os.path.join(CFD, "test_report.txt"), line_marker="*")
report.init()
report.add_hyperparameters()

print("\n".join(report.report)) 
