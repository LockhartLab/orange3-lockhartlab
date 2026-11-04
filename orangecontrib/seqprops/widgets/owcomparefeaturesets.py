import copy
import logging

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QBrush, QStandardItem

from Orange.base import Learner
from Orange.data import Table
from Orange.evaluation import Results
from Orange.preprocess.preprocess import Preprocess
from Orange.widgets import gui, settings
from Orange.widgets.evaluate.owtestandscore import results_one_vs_rest
from Orange.widgets.evaluate.utils import ScoreTable, scorer_caller, usable_scorers
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import Input, MultiInput, Output
from Orange.widgets.widget import Msg, OWWidget

from orangecontrib.seqprops import featuresets as fs

log = logging.getLogger(__name__)


def run(tables, learners, preprocessor, k, stratified, state: TaskState):
    def callback(progress):
        if state.is_interruption_requested():
            raise InterruptedError
        state.set_progress_value(100 * progress)
    # learners are not thread safe; test copies, like Test & Score does
    copies = [copy.deepcopy(learner) for learner in learners]
    return fs.evaluate(tables, copies, preprocessor, k, stratified, callback)


class OWCompareFeatureSets(OWWidget, ConcurrentWidgetMixin):
    name = "Compare Feature Sets"
    description = ("Cross-validate every model on every feature set "
                   "(for example several Select Columns) and score them in one table.")
    icon = "icons/CompareFeatureSets.svg"
    priority = 30
    keywords = "feature sets, compare, test and score, cross validation, select columns"

    class Inputs:
        data = MultiInput("Data", Table, doc="One feature set per connection.")
        learner = MultiInput("Learner", Learner)
        preprocessor = Input("Preprocessor", Preprocess)

    class Outputs:
        evaluations_results = Output("Evaluation Results", Results)

    class Error(OWWidget.Error):
        rows = Msg("{}")
        too_many_folds = Msg("There are fewer rows than folds.")
        failed = Msg("Something went wrong: {}")

    class Warning(OWWidget.Warning):
        cant_stratify = Msg("Cannot stratify: a class has fewer than {} rows.")
        learner_failed = Msg("{}")

    class Information(OWWidget.Information):
        need_inputs = Msg("Connect one or more feature sets (Data) and models (Learner).")

    NFolds = [2, 3, 5, 10, 20]
    TARGET_AVERAGE = "(None, show average over classes)"

    n_folds = settings.Setting(2)  # 5 folds, as in Test & Score
    class_selection = settings.Setting(TARGET_AVERAGE)
    score_table = settings.SettingProvider(ScoreTable)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.tables = []
        self.learners = []
        self.preprocessor = None
        self.results = None        # one Results per feature set
        self.scorers = []
        self._can_run = False
        self._watched_nodes = set()

        box = gui.vBox(self.controlArea, "Cross-validation")
        gui.comboBox(box, self, "n_folds", label="Number of folds:",
                     items=[str(k) for k in self.NFolds], orientation=Qt.Horizontal,
                     callback=self._invalidate)
        gui.widgetLabel(box, "Stratified, with the same folds for every set.")

        self.sets_box = gui.vBox(self.controlArea, "Feature sets")
        self.sets_label = gui.widgetLabel(self.sets_box, "")
        self.sets_label.setWordWrap(True)
        gui.rubber(self.controlArea)

        cbox = gui.hBox(self.mainArea)
        self.class_combo = gui.comboBox(
            cbox, self, "class_selection", label="Evaluation results for target",
            orientation=Qt.Horizontal, sendSelectedValue=True, contentsLength=25,
            callback=self._update_table)
        cbox.layout().addStretch(100)
        self.score_table = ScoreTable(self)
        self.score_table.shownScoresChanged.connect(self._update_table)
        self.mainArea.layout().addWidget(self.score_table.view)
        self._show_sets([])
        self.Information.need_inputs()

    # --- inputs -----------------------------------------------------------

    @Inputs.data
    def set_data(self, index, data):
        self.tables[index] = data

    @Inputs.data.insert
    def insert_data(self, index, data):
        self.tables.insert(index, data)

    @Inputs.data.remove
    def remove_data(self, index):
        self.tables.pop(index)

    @Inputs.learner
    def set_learner(self, index, learner):
        self.learners[index] = learner

    @Inputs.learner.insert
    def insert_learner(self, index, learner):
        self.learners.insert(index, learner)

    @Inputs.learner.remove
    def remove_learner(self, index):
        self.learners.pop(index)

    @Inputs.preprocessor
    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def handleNewSignals(self):
        tables = self._tables()
        class_var = tables[0].domain.class_var if tables else None
        self.scorers = usable_scorers(tables[0].domain) if class_var is not None else []
        self.score_table.update_header(self.scorers)
        selection = self.class_selection
        self.class_combo.clear()
        if class_var is not None and class_var.is_discrete:
            self.class_combo.addItems((self.TARGET_AVERAGE,) + class_var.values)
            keep = selection in class_var.values
            self.class_selection = selection if keep else self.TARGET_AVERAGE
        self._invalidate()

    # --- evaluation -------------------------------------------------------

    def _tables(self):
        return [t for t in self.tables if t is not None]

    def _learners(self):
        return [lr for lr in self.learners if lr is not None]

    def _invalidate(self):
        self.cancel()
        self.results = None
        self._can_run = False
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        tables, learners = self._tables(), self._learners()
        self._show_sets(tables)
        if not tables or not learners:
            self.Information.need_inputs()
            self._finish()
            return
        problem = fs.rows_problem(tables)
        if problem:
            self.Error.rows(problem)
            self._finish()
            return
        k = self.NFolds[self.n_folds]
        if len(tables[0]) < k:
            self.Error.too_many_folds()
            self._finish()
            return
        stratified = fs.can_stratify(tables[0], k)
        if not stratified and tables[0].domain.has_discrete_class:
            self.Warning.cant_stratify(k)
        self._can_run = True
        self._update_table()
        self.start(run, tables, learners, self.preprocessor, k, stratified)

    def on_done(self, results):
        self.results = results
        self._finish()

    def on_exception(self, ex):
        log.exception("Compare Feature Sets failed", exc_info=ex)
        self.Error.failed(f"{type(ex).__name__}: {ex}")
        self.results = None
        self._can_run = False
        self._finish()

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def _finish(self):
        self._update_table()
        self._send()

    # --- names --------------------------------------------------------------

    def set_names(self):
        """A name per feature set: the label of the widget it comes from on
        the canvas, else the table's name, else 'Set n'."""
        tables = self._tables()
        titles = self._canvas_titles()
        names = []
        for i, table in enumerate(tables, start=1):
            name = titles.get(id(table))
            if not name:
                others = [t.name for t in tables if t is not table]
                name = table.name if table.name and table.name not in others else f"Set {i}"
            names.append(name)
        for i, name in enumerate(names):  # keep names unique
            if names.count(name) > 1:
                names[i] = f"{name} ({names[:i + 1].count(name)})"
        return names

    def _canvas_titles(self):
        """Titles of the canvas widgets that send each data table."""
        try:
            manager = self.signalManager
            workflow = manager.workflow()
            node = workflow.widget_manager.node_for_widget(self)
            titles = {}
            for link in workflow.find_links(sink_node=node):
                if link.sink_channel.name != "Data":
                    continue
                for value in manager.link_contents(link).values():
                    if value is not None:
                        titles[id(value)] = link.source_node.title
                source = link.source_node
                if source not in self._watched_nodes:
                    self._watched_nodes.add(source)
                    source.title_changed.connect(self._names_changed)
            return titles
        except Exception:  # not on a canvas (scripts, tests)
            return {}

    def _names_changed(self, *_):
        self._show_sets(self._tables())
        self._update_table()
        self._send()

    def _show_sets(self, tables):
        if not tables:
            self.sets_label.setText("No feature sets connected.")
            return
        self.sets_label.setText("\n".join(
            f"{name}: {len(t.domain.attributes)} features"
            for name, t in zip(self.set_names(), tables)))

    # --- output -------------------------------------------------------------

    def _send(self):
        if not self.results:
            self.Outputs.evaluations_results.send(None)
            return
        names = [getattr(lr, "name", type(lr).__name__) for lr in self.results[0].learners]
        self.Outputs.evaluations_results.send(
            fs.combine(self.results, self.set_names(), names))

    def _update_table(self):
        model = self.score_table.model
        model.removeRows(0, model.rowCount())
        tables, learners = self._tables(), self._learners()
        if not self._can_run:
            return
        target = None
        class_var = tables[0].domain.class_var
        if class_var.is_discrete and self.class_selection in class_var.values:
            target = class_var.values.index(self.class_selection)
        errors = []
        for s, set_name in enumerate(self.set_names()):
            per_model = list(self.results[s].split_by_model()) if self.results else None
            for m, learner in enumerate(learners):
                name = fs.combined_name(getattr(learner, "name", type(learner).__name__), set_name)
                head = QStandardItem(name)
                row = [head, QStandardItem(), QStandardItem()]
                if per_model is None:
                    head.setText(f"{name} (running)")
                elif per_model[m].failed[0]:
                    head.setText(f"{name} (error)")
                    head.setForeground(QBrush(Qt.red))
                    head.setToolTip(str(per_model[m].failed[0]))
                    errors.append(f"{name}: {per_model[m].failed[0]}")
                else:
                    res = per_model[m]
                    if target is not None:
                        res, target_arg = results_one_vs_rest(res, target), 1
                    else:
                        target_arg = None
                    for scorer in self.scorers:
                        item = QStandardItem()
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        try:
                            value = scorer_caller(scorer, res, target=target_arg)()
                            item.setData(float(value[0]), Qt.DisplayRole)
                        except Exception as ex:  # a score that does not apply
                            item.setToolTip(str(ex))
                        row.append(item)
                model.appendRow(row)
        self.Warning.learner_failed("\n".join(errors), shown=bool(errors))
        self.score_table.view.resizeColumnsToContents()

    def send_report(self):
        self.report_items((
            ("Feature sets", ", ".join(self.set_names())),
            ("Cross-validation", f"{self.NFolds[self.n_folds]} folds, stratified"),
            ("Target class", self.class_selection.strip("()")),
        ))
        self.report_table("Scores", self.score_table.view)


if __name__ == "__main__":  # pragma: no cover
    from Orange.classification import LogisticRegressionLearner, SVMLearner
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    iris = Table("iris")
    petals, sepals = iris[:, ["petal length", "petal width", "iris"]], iris[:, [0, 1, "iris"]]
    petals.name, sepals.name = "petals", "sepals"
    WidgetPreview(OWCompareFeatureSets).run(
        insert_data=[(0, petals), (1, sepals)],
        insert_learner=[(0, LogisticRegressionLearner()), (1, SVMLearner())])
