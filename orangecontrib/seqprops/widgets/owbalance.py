from Orange.data import Table
from Orange.modelling import Learner
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.widget import Input, Output

from orangecontrib.seqprops.balance import Balance, BalancedLearner, DOWN, UP, METHOD_NAMES

METHOD_LABELS = {DOWN: "Downsample the larger class",
                 UP: "Upsample the smaller class"}


class OWBalance(OWBaseLearner):
    name = "Balance"
    description = ("Train models on class-balanced data: wrap one model, or "
                   "connect the Preprocessor output to Test & Score to balance "
                   "every model there. Only training data is resampled.")
    icon = "icons/Balance.svg"
    priority = 20
    keywords = ("balance, imbalance, imbalanced, class, resample, downsample, "
                "upsample, undersample, oversample")

    LEARNER = BalancedLearner

    learner_name = Setting("", schema_only=True)
    method = Setting(DOWN)

    class Inputs(OWBaseLearner.Inputs):
        base_learner = Input("Base Learner", Learner)

    class Outputs(OWBaseLearner.Outputs):
        # Not tied to a model: to Test & Score's Preprocessor input, it
        # balances the training part of each fold for all models there.
        preprocessor = Output("Preprocessor", Preprocess)

    def __init__(self):
        super().__init__()
        self.base_learner = None

    def add_main_layout(self):
        gui.radioButtons(
            self.controlArea, self, "method",
            [METHOD_LABELS[DOWN], METHOD_LABELS[UP]],
            box="Balance the training data", callback=self._method_changed)

    @Inputs.base_learner
    def set_learner(self, learner):
        self.base_learner = learner
        self._set_default_name()
        self.learner = self.model = None

    def _set_default_name(self):
        self.set_default_learner_name(
            "" if self.base_learner is None
            else f"{self.base_learner.name} ({METHOD_NAMES[self.method]})")

    def _method_changed(self):
        self._set_default_name()
        self.apply()

    def update_learner(self):
        super().update_learner()
        self.Outputs.preprocessor.send(Balance(self.method))

    def create_learner(self):
        if self.base_learner is None:
            return None
        return BalancedLearner(self.base_learner, self.method,
                               preprocessors=self.preprocessors)

    def get_learner_parameters(self):
        return (("Balancing", METHOD_LABELS[self.method]),)


if __name__ == "__main__":  # pragma: no cover
    from Orange.classification import SVMLearner
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWBalance).run(Table("heart_disease"), set_learner=SVMLearner())
