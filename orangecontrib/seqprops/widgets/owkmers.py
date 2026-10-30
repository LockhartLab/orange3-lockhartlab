import logging

from Orange.data import Domain, DiscreteVariable, StringVariable, Table
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.seqprops import kmers as km
from orangecontrib.seqprops.widgets.owsequenceproperties import sequence_columns

log = logging.getLogger(__name__)

MAX_K = 10
# More columns than this make Orange slow; warn but still compute.
MANY_COLUMNS = 10000


class OWKmers(OWWidget):
    name = "K-mer Bag of Words"
    description = ("Count the k-mers (overlapping substrings of length k) "
                   "in protein or DNA/RNA sequences: one column per k-mer.")
    icon = "icons/Kmers.svg"
    priority = 20
    keywords = "kmer, k-mer, bag of words, n-gram, composition, sequence"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    class Error(OWWidget.Error):
        no_sequence_column = Msg("Input has no text column with sequences.")
        unexpected = Msg("Something went wrong: {}")

    class Warning(OWWidget.Warning):
        many_columns = Msg("{} k-mer columns; Orange may be slow. "
                           "Try a smaller k.")

    settingsHandler = DomainContextHandler()
    sequence_var = ContextSetting(None)
    k_from = Setting(1)
    k_to = Setting(2)
    frequencies = Setting(True)
    auto_commit = Setting(True)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.data = None

        self.var_model = VariableListModel()
        gui.comboBox(self.controlArea, self, "sequence_var", box="Sequences",
                     label="Column:", orientation="horizontal",
                     model=self.var_model, callback=self.commit.deferred)

        box = gui.vBox(self.controlArea, "K-mers")
        row = gui.hBox(box)
        gui.spin(row, self, "k_from", 1, MAX_K, label="k from",
                 callback=self._k_from_changed)
        gui.spin(row, self, "k_to", 1, MAX_K, label="to",
                 callback=self._k_to_changed)
        gui.radioButtons(box, self, "frequencies",
                         ["Counts", "Frequencies (count ÷ number of k-mers)"],
                         callback=self.commit.deferred)

        gui.auto_apply(self.buttonsArea, self, "auto_commit")

    def _k_from_changed(self):
        self.k_to = max(self.k_to, self.k_from)
        self.commit.deferred()

    def _k_to_changed(self):
        self.k_from = min(self.k_from, self.k_to)
        self.commit.deferred()

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.Error.no_sequence_column.clear()
        self.data = data
        self.sequence_var = None
        candidates = sequence_columns(data.domain) if data is not None else []
        self.var_model[:] = candidates
        if data is not None:
            if not candidates:
                self.Error.no_sequence_column()
                self.data = None
            else:
                named = [v for v in candidates if "seq" in v.name.lower()]
                self.sequence_var = (named or candidates)[0]
                self.openContext(data.domain)
        self.commit.now()

    @gui.deferred
    def commit(self):
        self.Warning.clear()
        self.Error.unexpected.clear()
        try:
            out = self._output()
        except Exception as ex:  # show the problem in the widget, never crash
            log.exception("K-mer Bag of Words failed")
            self.Error.unexpected(f"{type(ex).__name__}: {ex}")
            out = None
        self.Outputs.data.send(out)

    def _output(self):
        if self.data is None or self.sequence_var is None:
            return None
        sequences = km.sequence_values(self.data, self.sequence_var)
        kmers = km.observed_kmers(sequences, self.k_from, self.k_to)
        if len(kmers) > MANY_COLUMNS:
            self.Warning.many_columns(len(kmers))
        domain = self.data.domain
        new_vars = km.kmer_variables(self.sequence_var, kmers,
                                     bool(self.frequencies), domain)
        out = self.data.transform(
            Domain(domain.attributes + tuple(new_vars), domain.class_vars, domain.metas))
        out.name = self.data.name
        return out

    def send_report(self):
        if self.data is None:
            return
        self.report_items((
            ("Sequence column", self.sequence_var.name),
            ("k", f"{self.k_from} to {self.k_to}"),
            ("Values", "frequencies" if self.frequencies else "counts"),
        ))


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    demo = Table.from_list(
        Domain([], DiscreteVariable("class", values=("amp", "toxin")),
               [StringVariable("sequence")]),
        [["toxin", "MKTAYIAKQRQISFVKSHFSRQ"], ["amp", "GIGKFLHSAKKFGKAFVGEIMNS"]])
    WidgetPreview(OWKmers).run(demo)
