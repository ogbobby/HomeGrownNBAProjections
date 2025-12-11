# from fuckme import SimpleNBAProjection
# from injurySrape import get_injuries


# model = SimpleNBAProjection("DKSalaries.csv")
# injuries = get_injuries()  # your ESPN injury scraper returns dict
# df = model.run(save_csv="projections.csv", injuries=injuries, n_sims=2000)
# print(df.head())

# run_projections.py
from fuckme import SimpleNBAProjection
from injurySrape import get_injuries  # your existing function

# 1️⃣ Load injuries from ESPN
injuries_dict = get_injuries()  # should return dict: {'player_name_lower': 'OUT/QUESTIONABLE/etc'}

# 2️⃣ Initialize the projection model
dk_csv_path = "/home/iamgeneral/Documents/git/HomeGrownNBAProjection/DKSalaries.csv"  # path to your DK CSV
model = SimpleNBAProjection(dk_csv_path)

# 3️⃣ Load DK salaries
if not model.load_dk_salaries():
    raise RuntimeError("Failed to load DraftKings salaries")

# 4️⃣ Fetch team stats and today's matchups
model.fetch_team_stats()
model.fetch_todays_matchups()

# 5️⃣ Run projections with injuries and Monte Carlo simulation
projections_df = model.run(
    save_csv='nba_projections.csv',  # optional: saves CSV
    injuries=injuries_dict,
    n_sims=2000  # number of Monte Carlo simulations
)

# 6️⃣ Display top 10 projected players
print(projections_df.head(10))
