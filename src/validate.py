import pandas as pd
import json
import os

def validate_campaigns(df):
    """
    Runs consistency checks on the unified feature table.
    Returns a dict with validation results and flagged issues.
    """
    issues = []
    summary = {}

    # --- Check 1: Zero spend with non-zero revenue ---
    bad_attribution = df[(df['spend'] == 0) & (df['revenue'] > 0)]
    if len(bad_attribution) > 0:
        issues.append({
            'check': 'zero_spend_nonzero_revenue',
            'severity': 'high',
            'count': len(bad_attribution),
            'detail': bad_attribution[['channel', 'campaign_name', 'date', 
                                       'spend', 'revenue']].to_dict('records')[:5]
        })
    summary['zero_spend_nonzero_revenue'] = len(bad_attribution)

    # --- Check 2: Extreme ROAS outliers ---
    extreme_roas = df[(df['roas'] > 50) & (df['spend'] > 0)]
    if len(extreme_roas) > 0:
        issues.append({
            'check': 'extreme_roas_outliers',
            'severity': 'medium',
            'count': len(extreme_roas),
            'detail': extreme_roas[['channel', 'campaign_name', 'date',
                                     'spend', 'revenue', 'roas']].to_dict('records')[:5]
        })
    summary['extreme_roas_outliers'] = len(extreme_roas)

    # --- Check 3: Thin history campaigns ---
    thin = df.groupby(['channel', 'campaign_name'])['date'].count()
    thin_campaigns = thin[thin < 14].reset_index()
    thin_campaigns.columns = ['channel', 'campaign_name', 'days_of_data']
    if len(thin_campaigns) > 0:
        issues.append({
            'check': 'thin_history_campaigns',
            'severity': 'low',
            'count': len(thin_campaigns),
            'detail': thin_campaigns.to_dict('records')
        })
    summary['thin_history_campaigns'] = len(thin_campaigns)

    # --- Check 4: Large date gaps per campaign ---
    gap_flags = []
    for (channel, campaign), grp in df.groupby(['channel', 'campaign_name']):
        grp = grp.sort_values('date')
        gaps = grp['date'].diff().dt.days
        large_gaps = gaps[gaps > 30]
        if len(large_gaps) > 0:
            gap_flags.append({
                'channel': channel,
                'campaign_name': campaign,
                'gap_days': int(large_gaps.max())
            })
    if gap_flags:
        issues.append({
            'check': 'large_date_gaps',
            'severity': 'medium',
            'count': len(gap_flags),
            'detail': gap_flags[:5]
        })
    summary['campaigns_with_large_gaps'] = len(gap_flags)

    # --- Check 5: Spend exceeding daily budget ---
    budget_breach = df[
        (df['daily_budget'] > 0) & 
        (df['spend'] > df['daily_budget'] * 1.2)
    ]
    if len(budget_breach) > 0:
        issues.append({
            'check': 'spend_exceeds_budget',
            'severity': 'low',
            'count': len(budget_breach),
            'detail': budget_breach[['channel', 'campaign_name', 'date',
                                      'spend', 'daily_budget']].to_dict('records')[:5]
        })
    summary['spend_exceeds_budget'] = len(budget_breach)

    return {
        'total_rows': len(df),
        'total_campaigns': df['campaign_name'].nunique(),
        'channels': df['channel'].unique().tolist(),
        'summary': summary,
        'issues': issues,
        'passed': len([i for i in issues if i['severity'] == 'high']) == 0
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    print("Running validation checks...")
    df = pd.read_parquet(args.features)
    report = validate_campaigns(df)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Validation complete. Passed: {report['passed']}")
    print(f"Summary: {report['summary']}")
    print(f"Report saved to {args.out}")


if __name__ == '__main__':
    main()