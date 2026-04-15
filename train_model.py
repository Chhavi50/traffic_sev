from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

def train_model(df):
    # Split features and target
    X = df.drop('most_severe_injury', axis=1)
    y = df['most_severe_injury']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 APPLY SMOTE HERE (after split)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, 'model.pkl')

    return model, X_test, y_test