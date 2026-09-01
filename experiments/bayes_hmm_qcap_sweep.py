# -*- coding: utf-8 -*-
"""Extended finite-window debt-cap frontier for Bayes-FreshSky-Safe."""
import os
import pandas as pd
import bayes_hmm_experiment as E
import belief_whittle as B


def main():
    summaries = []
    seeds = []
    for qcap in [4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]:
        cfg = E.clone_cfg(pbar=0.4, qcap=qcap)
        frame, summary = E.run_detailed(E.BayesFreshSkySafe, cfg)
        summary.update(policy="Bayes-FreshSky-Safe", qcap=qcap)
        summaries.append(summary)
        seeds.append(frame.assign(policy="Bayes-FreshSky-Safe", qcap=qcap))
        print(
            f"qcap={qcap:5.1f} W-AoI={summary['wAoI']:7.1f}"
            f"+/-{summary['wAoI_std']:5.1f} maxP={summary['max_avg_power']:.3f} "
            f"maxQ={summary['max_queue']:.3f} cert={summary['certificate_all_seeds']}"
        )
    os.makedirs(B.RES, exist_ok=True)
    pd.DataFrame(summaries).to_csv(
        os.path.join(B.RES, "bayes_hmm_qcap_summary.csv"), index=False
    )
    pd.concat(seeds, ignore_index=True).to_csv(
        os.path.join(B.RES, "bayes_hmm_qcap_seeds.csv"), index=False
    )
    print("saved bayes_hmm_qcap_summary.csv and bayes_hmm_qcap_seeds.csv")


if __name__ == "__main__":
    main()
