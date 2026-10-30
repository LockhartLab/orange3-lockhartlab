import numpy as np

from Orange.data import Domain, ContinuousVariable, DiscreteVariable, StringVariable, Table
from Orange.data.util import get_unique_names
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.seqprops import properties as sp

AUTO, PROTEIN, NUCLEOTIDE = range(3)
TYPE_LABELS = ["Detect automatically", "Protein (amino acids)", "DNA / RNA (nucleotides)"]


class OWSequenceProperties(OWWidget):
    name = "Sequence Properties"
    description = ("Compute physicochemical properties (molecular weight, pI, "
                   "hydropathy, GC content, CpG ratio, ...) of protein or "
                   "DNA/RNA sequences.")
    icon = "icons/SequenceProperties.svg"
    priority = 10
    keywords = "sequence, protein, peptide, amino acid, dna, rna, nucleotide, biopython"

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    class Error(OWWidget.Error):
        no_sequence_column = Msg("Input has no text column with sequences.")

    class Warning(OWWidget.Warning):
        invalid = Msg("{} sequence(s) had no valid letters; their values are missing.")

    settingsHandler = DomainContextHandler()
    sequence_var = ContextSetting(None)
    sequence_type = Setting(AUTO)
    include_properties = Setting(True)
    include_skew = Setting(True)
    auto_commit = Setting(True)

    want_main_area = False
    resizing_enabled = False

    def __init__(self):
        super().__init__()
        self.data = None

        self.var_model = VariableListModel()
        box = gui.vBox(self.controlArea, "Sequences")
        gui.comboBox(box, self, "sequence_var", label="Column:",
                     orientation="horizontal", model=self.var_model,
                     callback=self.commit.deferred)
        gui.radioButtons(box, self, "sequence_type", TYPE_LABELS,
                         callback=self.commit.deferred)

        box = gui.vBox(self.controlArea, "Output")
        gui.checkBox(box, self, "include_properties", "Physicochemical properties",
                     callback=self.commit.deferred)

        self.dna_box = gui.vBox(self.controlArea, "DNA / RNA only")
        gui.checkBox(self.dna_box, self, "include_skew", "GC/AT skew, CpG ratio",
                     callback=self.commit.deferred)

        self.info_label = gui.widgetLabel(gui.vBox(self.controlArea, "Info"),
                                          "No data on input.")

        gui.auto_apply(self.buttonsArea, self, "auto_commit")

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.Error.clear()
        self.data = data
        self.sequence_var = None
        candidates = self._sequence_columns(data.domain) if data is not None else []
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

    @staticmethod
    def _sequence_columns(domain):
        """Text columns, plus categorical columns whose values all look like
        sequences (so class labels such as 'Iris-setosa' are not offered)."""
        text = [v for v in domain.metas if isinstance(v, StringVariable)]
        categorical = [v for v in domain.variables + domain.metas
                       if isinstance(v, DiscreteVariable)
                       and all(sp.looks_like_sequence(sp.clean(x)) for x in v.values)]
        return text + categorical

    def _sequences(self):
        var = self.sequence_var
        column = self.data.get_column(var)
        if var.is_discrete:
            column = [var.values[int(v)] if not np.isnan(v) else "" for v in column]
        return [sp.clean(s) for s in column]

    @gui.deferred
    def commit(self):
        self.Warning.clear()
        if self.data is None or self.sequence_var is None:
            self.dna_box.setEnabled(True)
            self.info_label.setText("No data on input.")
            self.Outputs.data.send(None)
            return

        sequences = self._sequences()
        kind = {AUTO: sp.detect_type(sequences), PROTEIN: sp.PROTEIN,
                NUCLEOTIDE: sp.NUCLEOTIDE}[self.sequence_type]
        self.dna_box.setEnabled(kind == sp.NUCLEOTIDE)
        names, new_x, n_invalid = sp.compute(sequences, kind, self._groups())
        if n_invalid:
            self.Warning.invalid(n_invalid)

        domain = self.data.domain
        names = get_unique_names(domain, names)
        new_domain = Domain(list(domain.attributes) + [ContinuousVariable(n) for n in names],
                            domain.class_vars, domain.metas)
        out = Table.from_numpy(new_domain, np.hstack([self.data.X, new_x]),
                               self.data.Y, self.data.metas, self.data.W,
                               attributes=self.data.attributes)
        out.name = self.data.name

        label = "protein" if kind == sp.PROTEIN else "DNA/RNA"
        self.info_label.setText(f"{len(sequences)} {label} sequences\n"
                                f"{len(names)} columns added")
        self.Outputs.data.send(out)

    def _groups(self):
        return [group for group, on in ((sp.PROPERTIES, self.include_properties),
                                        (sp.SKEW, self.include_skew)) if on]

    def send_report(self):
        if self.data is None:
            return
        self.report_items((
            ("Sequence column", self.sequence_var.name),
            ("Sequence type", TYPE_LABELS[self.sequence_type]),
            ("Properties", self.include_properties),
            ("GC/AT skew, CpG ratio (DNA/RNA)", self.include_skew),
        ))


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    demo = Table.from_list(
        Domain([], DiscreteVariable("class", values=("amp", "toxin")),
               [StringVariable("sequence")]),
        [["toxin", "MKTAYIAKQRQISFVKSHFSRQ"], ["amp", "GIGKFLHSAKKFGKAFVGEIMNS"]])
    WidgetPreview(OWSequenceProperties).run(demo)
