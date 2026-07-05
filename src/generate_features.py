import argparse
import pandas as pd
import os

def load_google(data_dir):
    path = os.path.join(data_dir, 'google_ads_campaign_stats.csv')
    g = pd.read_csv(path, index_col=0)
    return pd.DataFrame({
        'date': pd.to_datetime(g['segments_date']),
        'channel': 'Google',
        'campaign_id': g['campaign_id'].astype(str),
        'campaign_name': g['campaign_name'],
        'campaign_type': g['campaign_advertising_channel_type'].str.title(),
        'spend': g['metrics_cost_micros'] / 1e6,
        'revenue': g['metrics_conversions_value'],
        'clicks': g['metrics_clicks'],
        'impressions': g['metrics_impressions'],
        'daily_budget': g['campaign_budget_amount'],
    })

def load_meta(data_dir):
    path = os.path.join(data_dir, 'meta_ads_campaign_stats.csv')
    m = pd.read_csv(path, index_col=0)
    def parse_type(name):
        prefix = name.split('_')[0]
        return {'Generic': 'Generic', 'Prospecting': 'Prospecting',
                'Remarketing': 'Remarketing'}.get(prefix, 'Unknown')
    return pd.DataFrame({
        'date': pd.to_datetime(m['date_start']),
        'channel': 'Meta',
        'campaign_id': m['campaign_id'].astype(str),
        'campaign_name': m['campaign_name'],
        'campaign_type': m['campaign_name'].apply(parse_type),
        'spend': m['spend'],
        'revenue': m['conversion'],
        'clicks': m['clicks'],
        'impressions': m['impressions'],
        'daily_budget': m['daily_budget'],
    })

def load_bing(data_dir):
    path = os.path.join(data_dir, 'bing_campaign_stats.csv')
    b = pd.read_csv(path, index_col=0)
    return pd.DataFrame({
        'date': pd.to_datetime(b['TimePeriod']),
        'channel': 'Bing',
        'campaign_id': b['CampaignId'].astype(str),
        'campaign_name': b['CampaignName'],
        'campaign_type': b['CampaignType'],
        'spend': b['Spend'],
        'revenue': b['Revenue'],
        'clicks': b['Clicks'],
        'impressions': b['Impressions'],
        'daily_budget': b['DailyBudget'],
    })

def build_features(df):
    # Compute ROAS
    df['roas'] = df.apply(
        lambda r: r['revenue'] / r['spend'] if r['spend'] > 0 else 0, axis=1
    )
    # Sort by campaign and date
    df = df.sort_values(['channel', 'campaign_name', 'date']).reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    print("Loading data...")
    google = load_google(args.data_dir)
    meta = load_meta(args.data_dir)
    bing = load_bing(args.data_dir)

    print("Combining channels...")
    combined = pd.concat([google, meta, bing], ignore_index=True)

    print("Building features...")
    features = build_features(combined)

    print(f"Feature table shape: {features.shape}")
    features.to_parquet(args.out, index=False)
    print(f"Features saved to {args.out}")

if __name__ == '__main__':
    main()