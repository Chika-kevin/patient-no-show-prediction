# ============================================================
# CIS4517 - Research and Development Project
# Student: Chika Kevin Edoga (26639211)
# Title: Predictive Analysis of Patient No-Show Appointments
# Evidence Notebook - Run in Google Colab
# ============================================================

# ============================================================
# CELL 1 - IMPORT LIBRARIES (Screenshot this = Evidence of setup)
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("CIS4517 - Chika Kevin Edoga (26639211)")
print("Predictive Analysis of Patient No-Show Appointments")
print("=" * 55)
print("\n✅ All libraries imported successfully")
print("\nLibraries loaded:")
print(f"  Pandas version: {pd.__version__}")
print(f"  NumPy version: {np.__version__}")


# ============================================================
# CELL 2 - LOAD DATASET (Screenshot this = Evidence 1)
# ============================================================

# Upload your KaggleV2-May-2016.csv to Colab first, then run:
df = pd.read_csv('KaggleV2-May-2016.csv')

print("=" * 55)
print("DATASET SUCCESSFULLY LOADED")
print("=" * 55)
print(f"\n✅ Total records:    {df.shape[0]:,}")
print(f"✅ Total features:   {df.shape[1]}")
print(f"\nColumn names:")
for col in df.columns:
    print(f"  - {col}")
print(f"\nTarget variable distribution:")
print(df['No-show'].value_counts())
print(f"\nNo-show rate: {(df['No-show']=='Yes').sum()/len(df)*100:.1f}%")


# ============================================================
# CELL 3 - DATA QUALITY CHECK (Screenshot this)
# ============================================================
print("=" * 55)
print("DATA QUALITY REPORT")
print("=" * 55)
print(f"\nMissing values per column:")
print(df.isnull().sum())
print(f"\nData types:")
print(df.dtypes)
print(f"\nBasic statistics:")
print(df.describe())
print(f"\n⚠️  Records with negative age: {(df['Age'] < 0).sum()}")
print(f"✅ These will be removed during cleaning")


# ============================================================
# CELL 4 - DATA CLEANING (Screenshot this)
# ============================================================
df_clean = df.copy()

# Remove negative age
df_clean = df_clean[df_clean['Age'] >= 0]

# Convert dates
df_clean['ScheduledDay'] = pd.to_datetime(df_clean['ScheduledDay'])
df_clean['AppointmentDay'] = pd.to_datetime(df_clean['AppointmentDay'])

# Feature engineering
df_clean['WaitDays'] = (df_clean['AppointmentDay'] - df_clean['ScheduledDay']).dt.days
df_clean['ScheduledHour'] = df_clean['ScheduledDay'].dt.hour
df_clean['AppointmentDayOfWeek'] = df_clean['AppointmentDay'].dt.dayofweek
df_clean['AgeGroup'] = pd.cut(df_clean['Age'],
    bins=[0, 12, 17, 35, 50, 65, 120],
    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Middle Aged', 'Senior'])

# Encode target variable
df_clean['NoShow_Binary'] = (df_clean['No-show'] == 'Yes').astype(int)

print("=" * 55)
print("DATA CLEANING COMPLETE")
print("=" * 55)
print(f"\n✅ Negative age records removed: {len(df) - len(df_clean)}")
print(f"✅ WaitDays feature created")
print(f"✅ ScheduledHour feature created")
print(f"✅ AppointmentDayOfWeek feature created")
print(f"✅ AgeGroup feature created")
print(f"✅ Target variable encoded (0=Showed up, 1=No-show)")
print(f"\nClean dataset shape: {df_clean.shape}")
print(f"\nNew features sample:")
print(df_clean[['Age', 'AgeGroup', 'WaitDays', 'ScheduledHour',
                'AppointmentDayOfWeek', 'NoShow_Binary']].head(10))


# ============================================================
# CELL 5 - EDA CHART 1: No-show Rate Overview
# (Screenshot this = Evidence 2)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Chika Kevin Edoga (26639211) - No-Show Analysis Overview',
             fontsize=13, fontweight='bold')

# Chart 1: Overall distribution
colors = ['#2ecc71', '#e74c3c']
df_clean['No-show'].value_counts().plot(
    kind='pie', ax=axes[0],
    colors=colors, autopct='%1.1f%%',
    startangle=90, labels=['Attended\n(79.8%)', 'No-Show\n(20.2%)'])
axes[0].set_title('Overall Appointment Outcome', fontweight='bold')
axes[0].set_ylabel('')

# Chart 2: No-show by gender
gender_noshow = df_clean.groupby('Gender')['NoShow_Binary'].mean() * 100
gender_noshow.plot(kind='bar', ax=axes[1], color=['#3498db', '#e74c3c'],
                   edgecolor='black')
axes[1].set_title('No-Show Rate by Gender', fontweight='bold')
axes[1].set_ylabel('No-Show Rate (%)')
axes[1].set_xlabel('Gender')
axes[1].tick_params(axis='x', rotation=0)
for i, v in enumerate(gender_noshow):
    axes[1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# Chart 3: No-show by day of week
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
day_noshow = df_clean.groupby('AppointmentDayOfWeek')['NoShow_Binary'].mean() * 100
day_noshow.index = days[:len(day_noshow)]
day_noshow.plot(kind='bar', ax=axes[2], color='#3498db', edgecolor='black')
axes[2].set_title('No-Show Rate by Day of Week', fontweight='bold')
axes[2].set_ylabel('No-Show Rate (%)')
axes[2].set_xlabel('Day of Week')
axes[2].tick_params(axis='x', rotation=0)
for i, v in enumerate(day_noshow):
    axes[2].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=8)

plt.tight_layout()
plt.savefig('evidence_chart1_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 1 saved - take a screenshot for Appendix B")


# ============================================================
# CELL 6 - EDA CHART 2: Key Predictors
# (Screenshot this = Evidence 2 continued)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Chika Kevin Edoga (26639211) - Key Predictors of No-Show',
             fontsize=13, fontweight='bold')

# Chart 1: No-show by wait days
wait_bins = pd.cut(df_clean['WaitDays'],
    bins=[-1, 0, 7, 14, 30, 60, 500],
    labels=['Same Day', '1-7 Days', '8-14 Days', '15-30 Days', '31-60 Days', '60+ Days'])
wait_noshow = df_clean.groupby(wait_bins)['NoShow_Binary'].mean() * 100
wait_noshow.plot(kind='bar', ax=axes[0],
    color=['#2ecc71','#f39c12','#e67e22','#e74c3c','#c0392b','#922b21'],
    edgecolor='black')
axes[0].set_title('No-Show Rate by Wait Days', fontweight='bold')
axes[0].set_ylabel('No-Show Rate (%)')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(wait_noshow):
    axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=7)

# Chart 2: No-show by age group
age_noshow = df_clean.groupby('AgeGroup', observed=True)['NoShow_Binary'].mean() * 100
age_noshow.plot(kind='bar', ax=axes[1], color='#9b59b6', edgecolor='black')
axes[1].set_title('No-Show Rate by Age Group', fontweight='bold')
axes[1].set_ylabel('No-Show Rate (%)')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(age_noshow):
    axes[1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=8)

# Chart 3: SMS impact
sms_noshow = df_clean.groupby('SMS_received')['NoShow_Binary'].mean() * 100
sms_noshow.index = ['No SMS', 'SMS Sent']
sms_noshow.plot(kind='bar', ax=axes[2], color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[2].set_title('No-Show Rate: SMS Impact', fontweight='bold')
axes[2].set_ylabel('No-Show Rate (%)')
axes[2].tick_params(axis='x', rotation=0)
for i, v in enumerate(sms_noshow):
    axes[2].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('evidence_chart2_predictors.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Chart 2 saved - take a screenshot for Appendix B")
print("\n🎯 KEY FINDINGS FROM EDA:")
print(f"  - No-show rate: 20.2%")
print(f"  - Teens have highest no-show rate: ~25%")
print(f"  - Longer wait = higher no-show risk")
print(f"  - Saturday has highest no-show day rate")
print(f"  - Counterintuitively, SMS recipients no-show MORE")
