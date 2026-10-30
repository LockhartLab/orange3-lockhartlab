import logging

import numpy as np
from AnyQt.QtWidgets import QButtonGroup, QRadioButton

from Orange.data import Domain, ContinuousVariable, DiscreteVariable, StringVariable, Table
from Orange.data.util import get_unique_names
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import Input, Msg, Output, OWWidget

from orangecontrib.seqprops import properties as sp

log = logging.getLogger(__name__)

NOT_CHOSEN, PROTEIN, NUCLEOTIDE = -1, 0, 1
TYPE_LABELS = ["Protein (amino acids)", "DNA / RNA (nucleotides)"]
KINDS = {PROTEIN: sp.PROTEIN, NUCLEOTIDE: sp.NUCLEOTIDE}

# With DNA/RNA selected, more letters than this that cannot be nucleotides
# means the column almost certainly holds proteins.
MAX_NON_NUCLEOTIDE = 0.1


def _rows(indices, limit=5):
    """1-based row numbers, as the Data Table shows them."""
    shown = ", ".join(str(i + 1) for i in indices[:limit])
    return shown + (", ..." if len(indices) > limit else "")


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

    class Information(OWWidget.Information):
        choose_type = Msg("Choose whether the sequences are protein or DNA / RNA.")

    class Error(OWWidget.Error):
        no_sequence_column = Msg("Input has no text column with sequences.")
        protein_as_dna = Msg("These look like protein sequences, but DNA / RNA "
                             "is selected. Choose Protein instead.")
        unexpected = Msg("Something went wrong: {}")

    class Warning(OWWidget.Warning):
        dna_as_protein = Msg("These look like DNA / RNA sequences, but Protein "
                             "is selected. Is that right?")
        empty = Msg("{} sequence(s) had no usable letters (row {}); "
                    "their values are blank.")
        failed = Msg("{} sequence(s) could not be processed (row {}: {}); "
                     "their values are blank.")

    settingsHandler = DomainContextHandler()
    settings_version = 2
    sequence_var = ContextSetting(None)
    # Saved with the workflow, but never remembered as the default for new
    # widgets: every new widget asks which kind of sequence it gets.
    sequence_type = Setting(NOT_CHOSEN, schema_only=True)
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
        # Built by hand because gui.radioButtons always selects a button.
        self.type_buttons = QButtonGroup(self)
        for i, label in enumerate(TYPE_LABELS):
            button = QRadioButton(label)
            box.layout().addWidget(button)
            self.type_buttons.addButton(button, i)
        self.type_buttons.buttonClicked.connect(self._on_type_clicked)
        self._sync_type_buttons()

        gui.auto_apply(self.buttonsArea, self, "auto_commit")

    def _sync_type_buttons(self):
        """Show the current `sequence_type`, including 'nothing selected'."""
        if self.sequence_type in KINDS:
            self.type_buttons.button(self.sequence_type).setChecked(True)
        else:
            self.type_buttons.setExclusive(False)
            for button in self.type_buttons.buttons():
                button.setChecked(False)
            self.type_buttons.setExclusive(True)

    def _on_type_clicked(self, button):
        self.sequence_type = self.type_buttons.id(button)
        self._sync_type_buttons()
        self.commit.deferred()

    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.Error.no_sequence_column.clear()
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
        self.Information.clear()
        self.Warning.clear()
        self.Error.protein_as_dna.clear()
        self.Error.unexpected.clear()
        try:
            out = self._output()
        except Exception as ex:  # show the problem in the widget, never crash
            log.exception("Sequence Properties failed")
            self.Error.unexpected(f"{type(ex).__name__}: {ex}")
            out = None
        self.Outputs.data.send(out)

    def _output(self):
        if self.data is None or self.sequence_var is None:
            return None
        if self.sequence_type not in KINDS:
            self.Information.choose_type()
            return None

        sequences = self._sequences()
        kind = KINDS[self.sequence_type]
        if kind == sp.NUCLEOTIDE \
                and sp.non_nucleotide_fraction(sequences) > MAX_NON_NUCLEOTIDE:
            self.Error.protein_as_dna()
            return None
        if kind == sp.PROTEIN and sp.detect_type(sequences) == sp.NUCLEOTIDE:
            self.Warning.dna_as_protein()

        names, new_x, empty, failed = sp.compute(sequences, kind)
        if empty:
            self.Warning.empty(len(empty), _rows(empty))
        if failed:
            rows, reasons = zip(*failed)
            self.Warning.failed(len(failed), _rows(list(rows)), reasons[0])

        domain = self.data.domain
        names = get_unique_names(domain, names)
        new_domain = Domain(list(domain.attributes) + [ContinuousVariable(n) for n in names],
                            domain.class_vars, domain.metas)
        out = Table.from_numpy(new_domain, np.hstack([self.data.X, new_x]),
                               self.data.Y, self.data.metas, self.data.W,
                               attributes=self.data.attributes)
        out.name = self.data.name
        return out

    def send_report(self):
        if self.data is None:
            return
        self.report_items((
            ("Sequence column", self.sequence_var.name),
            ("Sequence type", TYPE_LABELS[self.sequence_type]
             if self.sequence_type in KINDS else "not chosen"),
        ))

    @classmethod
    def migrate_settings(cls, settings, version):
        if version is None or version < 2:
            # Version 1 had 0 = detect automatically, 1 = protein, 2 = DNA/RNA.
            old = settings.get("sequence_type")
            settings["sequence_type"] = {1: PROTEIN, 2: NUCLEOTIDE}.get(old, NOT_CHOSEN)


if __name__ == "__main__":  # pragma: no cover
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    demo = Table.from_list(
        Domain([], DiscreteVariable("class", values=("amp", "toxin")),
               [StringVariable("sequence")]),
        [["toxin", "MKTAYIAKQRQISFVKSHFSRQ"], ["amp", "GIGKFLHSAKKFGKAFVGEIMNS"]])
    WidgetPreview(OWSequenceProperties).run(demo)
