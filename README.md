# Same Fault, Two Valves

**LAB 01 · An experiment on a public benchmark**

![Valve one: alarm within 9 minutes. Valve two: no different from a normal day.](docs/same-fault-two-valves.png)

Two cooling-water valves start sticking at 08:00 on a simulated chemical plant, the Tennessee
Eastman process. A standard PCA process monitor raises an alarm on one within 9 minutes. The
other, it cannot tell apart from a normal day.

**Read the write-up, with interactive charts:** https://raymondcapisinio.com/lab/same-fault-two-valves/

This repository is the full pipeline behind it: download, measurement, an independent re-check,
and the page build. Every number in the write-up comes from this code.

## The result

The same fault (a sticking valve, "stiction") on two valves, 20 test runs each, scored from 08:00
to the end of the two-day run.

| | Samples flagged after 08:00 | Runs that raised an alarm |
|---|---:|---:|
| Valve one: reactor cooling water (fault 14) | 99.98% | 20 of 20 |
| Valve two: condenser cooling water (fault 15) | 2.34% | 8 of 20 |
| The same 20 runs with no fault (their twins) | | 5 of 20 |
| Normal running, 500 unseen runs | 2.05% | |

Every faulty test run in this dataset has a normal twin: it matches a fault-free run value for
value until 08:00, when the fault switches on. That makes a clean comparison: the same day, with
and without the fault. `06_export_page.py` checks the twins before it exports anything.

## How it compares

On valve one, every study that reports it catches 99.5 to 100% of samples after the fault. Valve two
splits them: plain PCA stays close to its own normal-day rate and one window-statistics study did not detect
it, but methods built for it reach over 95% (Shang and Chen, 2018) and 100% (Agarwal and colleagues, 2020,
with a network trained on labelled examples of the fault and a setpoint nudged every 4 h). Alarm limits
differ between studies and some used their own simulations, so read each row across. Published figures are
as reported in their papers. Rows marked **Mine** are this repo's checks, scored on 480 runs they never saw.

| Who | Whose method | Method | Valve two: readings flagged after the fault | Flagged on normal data | Note |
|---|---|---|---:|---:|---|
| [Agarwal and colleagues (2020)](https://arxiv.org/abs/2012.03861) | Published study | Deep neural network (LSTM) | 100% | 2.41% (all faults) | Trained on labelled examples of the fault, and nudges a setpoint every 4 h so it shows |
| [Shang and Chen (2018)](https://search.napstic.cn/literature/periodical/0220180600226670) | Published study | Dynamic transformed-component statistics | over 95% | not in the abstract | Figure from the paper's abstract |
| [Du and Du (2018)](https://skoge.folk.ntnu.no/prost/proceedings/ADCHEM-2018/media/files/0146.pdf) | Published study | Filtered signals + accumulating alarm (CUSUM) | detected, after 30 h on average | not given | Their own simulation (6-minute samples) |
| [Zhao (2017)](https://arxiv.org/abs/1712.04118) | Published study | PCA (SPE) | 25.6% | 16.9% | Many false alarms |
| [Lu and colleagues (2018)](https://web.mit.edu/braatzgroup/Lu_JProcessControl_2018.pdf) | Published study | Sparse canonical variate analysis | 25.5% | not given |  |
| [Chiang and Braatz (2003)](https://web.mit.edu/braatzgroup/54_Process_monitoring_using_causal_map_and_multivariate_statistics_fault_detection_and_identification.pdf) | Published study | Causal map + statistics | 24.9% | 4.62% | Calls this fault unobservable |
| [Sun and colleagues (2020)](https://arxiv.org/abs/1911.04386) | Published study | PCA (12 components) | 10.62% | 5.00% |  |
| [Sun and colleagues (2020)](https://arxiv.org/abs/1911.04386) | Published study | Bayesian recurrent neural network | 7.12% | 4.75% |  |
| [Peng, Zhao and Hu (2021)](https://arxiv.org/abs/2103.07303) | Published study | PCA | 5.75% | 1.88% |  |
| [Zhao (2017)](https://arxiv.org/abs/1712.04118) | Published study | PCA (T²) | 4.9% | 0.00% |  |
| [Galicia, He and Wang (2012)](https://skoge.folk.ntnu.no/prost/proceedings/cpc8-focapo-2012/data/papers/055.pdf) | Published study | Window statistics (SPA) | not detected | n/a | Their own simulation; PCA and dynamic PCA missed it too |
| **This repo**, 20 runs | Textbook method, run here | PCA (31 components) | 2.34% | 2.05% | No different from a normal day |
| **This repo**, 480 unseen runs | **Mine** | One reading's jitter, 24 h | 30.93% | 0.88% | Typical alarm after 18 h |
| **This repo**, 480 unseen runs | **Mine** | One reading's jitter, 8 h | 14.26% | 0.77% | Typical alarm after 13 h |
| **This repo**, 480 unseen runs | **Mine** | One reading, accumulating alarm (CUSUM) | 44.54% | 2.29% | Typical alarm after 14 h |
| **This repo**, 480 unseen runs | **Mine** | Four readings' jitter, 8 h | 15.97% | 0.92% | Typical alarm after 12 h |
| **This repo**, 480 unseen runs | **Mine** | All 52 readings' jitter, 12 h | 3.05% | 1.93% | Typical alarm after 19 h |

## A check that catches valve two

`05b_condenser_check.py` asks one question of one reading: has the condenser cooling-water outlet
temperature, XMEAS(22), been jumpier than normal over the last 24 hours? Jitter is the variance of
its sample-to-sample changes in that window; the limit is the level 99% of the normal training
windows stay below; the alarm rule is the same three in a row.

I designed it after looking at fault runs 1 to 20, so it is tested on runs 21 to 500 of the same
fault, which were not opened until it was fixed, each against its own normal twin:

| Unseen runs 21 to 500 | Whose method | Faulty runs that alarmed (higher is better) | Their twins that alarmed (lower is better) | Normal runs that alarmed after 08:00 | Readings flagged after the fault | Typical time to alarm |
|---|---|---:|---:|---:|---:|---:|
| Standard monitor (PCA) | Textbook method, run here | 147 of 480 (31%) | 131 of 480 (27%) | 136 of 500 | 2.4% | no better than its twins |
| One reading's jitter, 24 h | **Mine** | 429 of 480 (89%) | 38 of 480 (8%) | 40 of 500 | 30.9% | 18 h |
| One reading's jitter, 8 h | **Mine** | 421 of 480 (88%) | 83 of 480 (17%) | 85 of 500 | 14.3% | 13 h |
| One reading, accumulating alarm (CUSUM) | **Mine** | 441 of 480 (92%) | 85 of 480 (18%) | 87 of 500 | 44.5% | 14 h |
| Four readings' jitter, 8 h | **Mine** | 432 of 480 (90%) | 90 of 480 (19%) | 93 of 500 | 16.0% | 12 h |
| All 52 readings' jitter, 12 h | **Mine** | 229 of 480 (48%) | 178 of 480 (37%) | 186 of 500 | 3.0% | 19 h |

Three limits. It is slow: it needs most of a day of data. It watches one loop, so it has to be
told which loop to watch. And it says the loop got jumpier, not why: it also alarms in at least
half the runs of faults 2, 12, 17, 18 and 20. A once-a-day loop-health check, not an alarm.

## Three more tries

`05c_more_checks.py`: three ways to do better, designed the same way (settings fixed on runs 1-20,
limits from normal runs only, one test on the unseen runs): the jitter of **four linked readings**
over 8 h; an **accumulating alarm (CUSUM)** on the one reading; and the jitter of **all 52 readings**
over 12 h fed to a PCA monitor of its own.

| Unseen runs 21 to 500 | Whose method | Faulty runs that alarmed (higher is better) | Their twins that alarmed (lower is better) | Normal runs that alarmed after 08:00 | Readings flagged after the fault | Typical time to alarm |
|---|---|---:|---:|---:|---:|---:|
| Standard monitor (PCA) | Textbook method, run here | 147 of 480 (31%) | 131 of 480 (27%) | 136 of 500 | 2.4% | no better than its twins |
| One reading's jitter, 24 h | **Mine** | 429 of 480 (89%) | 38 of 480 (8%) | 40 of 500 | 30.9% | 18 h |
| One reading's jitter, 8 h | **Mine** | 421 of 480 (88%) | 83 of 480 (17%) | 85 of 500 | 14.3% | 13 h |
| One reading, accumulating alarm (CUSUM) | **Mine** | 441 of 480 (92%) | 85 of 480 (18%) | 87 of 500 | 44.5% | 14 h |
| Four readings' jitter, 8 h | **Mine** | 432 of 480 (90%) | 90 of 480 (19%) | 93 of 500 | 16.0% | 12 h |
| All 52 readings' jitter, 12 h | **Mine** | 229 of 480 (48%) | 178 of 480 (37%) | 186 of 500 | 3.0% | 19 h |

The accumulating alarm is the strongest of mine (more faulty runs, more readings flagged, sooner),
but it doubles the false alarms, which the 20 design runs did not show. Four readings caught about as
many as one, with more false alarms; all 52 barely separated faulty runs from twins. None comes near
the methods built for this fault.

## Try to beat it

Valve two is the hard one. On its 20 runs and their 20 normal twins, the PCA monitor alarms on
8 of the faulty runs and 5 of the twins, which is almost no separation. The checks above
separate them, but slowly, with false alarms, and only because they are pointed at the right loop.
Published methods built for this fault do much better (see the table). If you separate the 480
unseen runs from their twins faster, with fewer false alarms, or without being told where to look,
I would like to see it: open an issue, or write to hello@raymondcapisinio.com.

## Run it yourself

About 2 minutes after the downloads, under 2 GB of memory at peak (the re-check), no paid services.
Tested on Python 3.14 on Windows.

```bash
python -m pip install -r requirements.txt
cd pipeline
python 01_download.py fault-free        # 72 MB, checksum-verified, licence text re-read
python 01_download.py faulty-testing    # 837 MB
python 02_extract.py --check            # own streaming reader vs the rdata reference reader
python 02_extract.py                    # -> data/parquet/ (incl. the 480 unseen fault-15 runs)
python 03_measure.py                    # -> out/measurements.csv, out/monitor.json
python 05_sensitivity.py                # -> out/sensitivity.csv
python 05b_condenser_check.py           # -> out/condenser_check.json
python 05c_more_checks.py               # -> out/more_checks.json
python 06_export_page.py                # -> out/page_data.json
python 04_recheck.py                    # a separate implementation of every number; must PASS
python 07_build_page.py                 # -> page/index.html
python 09_og_card.py                    # -> page/og-card.png (needs: python -m playwright install chromium)
```

The results from my run are already in `out/`, so you can check them against yours.
`08_render_check.py` screenshots the page in a browser; the page takes its styling from the
website it is published on, which is not part of this repository.

## How the numbers are kept honest

- **Nothing typed by hand.** `07_build_page.py` fills every number in the write-up from `out/`.
- **Sentences are tested.** The build refuses to finish if the data no longer supports a sentence
  the write-up makes (for example "all 20 runs alarmed").
- **Measured twice.** `04_recheck.py` recomputes every quoted number a different way (SVD instead
  of an eigendecomposition, rolling windows instead of a loop, group-bys instead of arrays).
- **The reader is checked.** The 837 MB R data file is read by a small streaming reader written
  for this (`rstream.py`). `02_extract.py --check` compares it with the `rdata` reference reader,
  value for value, on both normal-running files.
- **The tuning is swept.** `05_sensitivity.py` reruns the monitor at 80, 90 and 95% of the
  variance and at a 99.9% limit. The picture does not change.
- **The new checks are tested on unseen runs.** Every check of mine was designed on fault runs
  1 to 20 and is scored on runs 21 to 500, which `02_extract.py` writes to a separate file.

## The monitor

PCA on the 52 standardised recorded values (41 measurements, 11 controller outputs), fitted to
250,000 samples of normal operation: 31 components (90.1% of the variance), Hotelling T² and
SPE limits at 99%. A sample is flagged when either statistic crosses its limit. An alarm is three
flags in a row, timed at the third. On normal running it flags 1.95% of samples.

## Layout

```
pipeline/   01-09 (with 05b and 05c, the extra checks) in the order above; rstream.py (R file reader),
            tags.py (names), probe_fault15.py
out/        measured results (CSV and JSON)
page/       template.html, the write-up with the placeholders that 07 fills
docs/       the image above
```

## How this was made

I wrote the pipeline with an AI coding assistant, to my specification. Every number comes from
that code and its separate re-check.

## Not claimed

This is a simulation, not a plant. It tests one standard monitor and four simple checks of mine, each
designed after seeing part of the data, then tested on runs it had not seen. None is the best published
result for this fault. Other studies' figures are as reported in their papers, not re-run here. The two field notes in the
write-up are fiction, and labelled as such. Neither the dataset's authors nor Pacific Science &
Engineering Group reviewed or endorse this work.

## Sources

- Data: Rieth, Amsel, Tran, Cook (2017), *Additional Tennessee Eastman Process Simulation Data
  for Anomaly Detection Evaluation*, Harvard Dataverse, [doi:10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1).
  Public domain. Not included here: `01_download.py` fetches it and verifies each file's checksum.
- Process: Downs and Vogel (1993), *A plant-wide industrial process control problem*, Computers
  & Chemical Engineering 17(3).
- Names of the recorded values and faults: N. L. Ricker, [Tennessee Eastman Challenge Archive](https://depts.washington.edu/control/LARRY/TE/download.html).
- Fault classes: Sun, Paiva, Xu, Sundaram, Braatz (2020), [arXiv:1911.04386](https://arxiv.org/abs/1911.04386).
- Faults most methods miss: Du and Du (2018), [ADCHEM](https://skoge.folk.ntnu.no/prost/proceedings/ADCHEM-2018/media/files/0146.pdf).
- Comparison table: Chiang and Braatz (2003), *Process monitoring using causal map and
  multivariate statistics*, Chemometrics and Intelligent Laboratory Systems 65 ([PDF](https://web.mit.edu/braatzgroup/54_Process_monitoring_using_causal_map_and_multivariate_statistics_fault_detection_and_identification.pdf));
  Zhao (2017), [arXiv:1712.04118](https://arxiv.org/abs/1712.04118); Lu, Jiang, Gopaluni, Loewen,
  Braatz (2018), Journal of Process Control 71 ([PDF](https://web.mit.edu/braatzgroup/Lu_JProcessControl_2018.pdf));
  Peng, Zhao, Hu (2021), Journal of Process Control 108, [arXiv:2103.07303](https://arxiv.org/abs/2103.07303);
  Agarwal, Mireles Gonzalez, Elkamel, Budman (2020), [arXiv:2012.03861](https://arxiv.org/abs/2012.03861);
  Shang and Chen (2018), IEEE Transactions on Industrial Electronics ([abstract](https://search.napstic.cn/literature/periodical/0220180600226670));
  Galicia, He, Wang (2012), FOCAPO ([PDF](https://skoge.folk.ntnu.no/prost/proceedings/cpc8-focapo-2012/data/papers/055.pdf)).

## Licence

Code: MIT, see [LICENSE](LICENSE). The dataset keeps its own terms (public domain).

Raymond P. Capisinio · [raymondcapisinio.com](https://raymondcapisinio.com)
