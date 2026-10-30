# Orange3 Sequence Properties

An add-on for the [Orange](https://orangedatamining.com/) data mining app.
It adds two widgets in a **Lockhart Lab** section:

- **Sequence Properties** turns protein or DNA/RNA sequences into numbers
  you can plot and use for machine learning: molecular weight, isoelectric
  point, hydropathy, GC content, melting temperature, and more.
- **Balance** makes any model train on class-balanced data, without
  spoiling Test & Score's cross-validation.

---

## Install it

You need **Orange 3.34 or newer** and an **internet connection**.
If you don't have Orange yet, download it from
[orangedatamining.com/download](https://orangedatamining.com/download/).

**1. Download the add-on file.**
Go to the
[**latest release**](https://github.com/LockhartLab/orange3-sequence-properties/releases/latest)
page. Under **Assets**, click the file whose name ends in **`.whl`**
(for example `orange3_sequence_properties-0.7.0-py3-none-any.whl`).
Don't rename it.

**2. Open the Add-ons window.**
Start Orange and choose **Options → Add-ons…** from the menu.

**3. Drag the file in.**
Drag the `.whl` file from your Downloads folder onto the Add-ons window.
A line called **Orange3-Sequence-Properties** appears with its box
checked. Click **OK** and wait until it finishes.

**4. Restart Orange.**
Quit Orange completely and open it again.

**5. Check it worked.**
The widget list on the left now has a **Lockhart Lab** section with the
**Sequence Properties** and **Balance** widgets in it. 🎉

### If something goes wrong

| Problem | Fix |
|---|---|
| No **Lockhart Lab** section after restarting | Make sure you *quit* Orange (on a Mac: **Orange → Quit Orange**), not just closed the window, then open it again. |
| The install fails with an error | Check you're online: the add-on also downloads Biopython while it installs. Then try again. |
| Nothing happens when you drop the file | Make sure the file name still ends in `.whl` and has no extra text like `(1)`. Delete duplicates and download it again. |
| "Permission denied" or similar (often on school computers) | Orange was installed for all users. Ask whoever manages the computer, or reinstall Orange just for your account. |
| Orange is older than 3.34 | Download and install the newest Orange first. |

**Updating:** download the newer `.whl` file and drag it in the same way.
**Removing:** in **Options → Add-ons…**, uncheck *Orange3-Sequence-Properties* and click **OK**.

---

## Use it

1. Put a **File** widget on the canvas and open a CSV file with a column
   of sequences (ideally named `sequence`).
2. Connect **File → Sequence Properties**, open the widget, and choose
   **Protein** or **DNA / RNA**. (Nothing comes out until you choose.)
3. Connect **Sequence Properties → Data Table** to see the new columns.
4. Try **Scatter Plot**, **Box Plot**, or **Tree → Test & Score** next.

Example files to try (on each file's GitHub page, click the
**Download raw file** button):

- [`examples/example_peptides.csv`](examples/example_peptides.csv): 12 protein sequences (peptides)
- [`examples/example_dna.csv`](examples/example_dna.csv): 14 short DNA sequences

Tips: if the choice doesn't match the sequences (for example DNA / RNA
selected for proteins), the widget tells you instead of producing nonsense.
Lower-case letters, spaces and a trailing `*` are fine; letters that aren't
standard residues/bases (X, B, Z, N, ...) are skipped. A sequence that can't
be processed gets blank values, and the widget names its row.

### Balance: when one class is much bigger than the other

With, say, three times more of one class, many models learn to predict the
big class almost every time. Accuracy still looks good, but the model finds
almost none of the small class.

1. Connect any model to **Balance**, for example **SVM → Balance**.
2. Choose **Downsample the larger class** (the default) or **Upsample the
   smaller class**.
3. Connect **Balance → Test & Score**, next to the plain model, and compare
   the *Recall* of the small class.

Balance resamples only the data the model is trained on. In Test & Score
that is the training part of each cross-validation fold, so the test rows
keep their real class ratio and no copied row ends up being tested.
Don't resample the data *before* Test & Score instead: copies then land in
the test folds and the scores look better than they are.
Balance also has a **Data** input and a **Model** output, for use with
**Predictions**.

## What Sequence Properties computes

| Protein (11 columns) | DNA / RNA (7 columns) |
|---|---|
| length | length |
| molecular weight | GC % |
| isoelectric point | molecular weight |
| charge at pH 7 | melting temp (C) |
| GRAVY (hydropathy) | GC skew |
| aromaticity | AT skew |
| instability index | CpG obs/exp |
| helix / turn / sheet fraction | |
| extinction coeff (reduced) | |

All values come from [Biopython](https://biopython.org/). The helix/turn/sheet
fractions are letter counts (tendencies), not structure predictions.

---

## For teachers and maintainers

**Make a new release** (this is what students download):

1. Change `version` in `pyproject.toml`, e.g. to `0.7.0`, and commit.
2. Tag and push: `git tag v0.7.0 && git push origin main v0.7.0`

GitHub Actions then tests the add-on on Windows and macOS, builds the
`.whl` file, and attaches it to a new release. The student link above always
points to the newest release.

**Build and test locally:**

    python -m pip install build
    python -m build --wheel --outdir dist .
    python -m pip install dist/*.whl PyQt6 PyQt6-WebEngine
    QT_QPA_PLATFORM=offscreen python -m unittest \
        orangecontrib.seqprops.widgets.tests.test_properties \
        orangecontrib.seqprops.widgets.tests.test_owsequenceproperties \
        orangecontrib.seqprops.widgets.tests.test_owbalance

## License

MIT. See [LICENSE](LICENSE).
