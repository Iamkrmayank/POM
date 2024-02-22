import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

# Load your dataset
df = pd.read_csv('/Users/kumarmayank/Downloads/Dataset_modified_pom.csv')
df = df.drop(['Product Availibility index', 'City'], axis=1)

# Assuming df is your DataFrame with 'Date', 'MRP', 'SP', and 'MSP' columns
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Features (X) and target variables (y)
X = df[['Year', 'Month', 'Day']]
y_mrp = df['MRP']
y_sp = df['SP']
y_msp = df['MSP']

# Split the data into training and testing sets
X_train, X_test, y_mrp_train, y_mrp_test, y_sp_train, y_sp_test, y_msp_train, y_msp_test = train_test_split(
    X, y_mrp, y_sp, y_msp, test_size=0.2, random_state=42
)

# Today's Tab

# Adding the seller name and other things
st.header("Seller Name:Ashok")
st.caption("Optimise your sales with Real-time online retail market analytics")

st.subheader("Dashboard")

# Create tabs using st
selected_tab = st.radio("Select Tab", ['Today', 'History/Forecast'])

if selected_tab == 'Today':
    # # Today's Tab Content
    selected_date = st.date_input("Select Date")
    selected_date_str = selected_date.strftime('%d-%m-%Y')  # Convert to 'DD-MM-YYYY' format

    # Filter data based on the selected date
    filtered_df = df[df['Date'].dt.strftime('%d-%m-%Y') == selected_date_str]

    if not filtered_df.empty:
        # Display the selected date
        st.write(f"Selected Date: {selected_date_str}")

        st.write(f"MSP:{float(filtered_df['MSP'].values)}")
        st.write(f"MRP:{float(filtered_df['MRP'].values)}")
        
        # Price range sliders
        min_price, max_price = st.slider("Price Range", float(filtered_df['MRP'].min()), float(filtered_df['MSP'].max()), (float(filtered_df['MRP'].min()), float(filtered_df['MSP'].max())))
        
        value = st.text_area("Enter the value between the above range", value=2.0)
        
        value = float(value)
        
        # Filter data based on selected price range
        filtered_price_range_df = filtered_df[(filtered_df['MRP'] >= min_price) & (filtered_df['MSP'] <= max_price)]

        if not filtered_price_range_df.empty:
            # Dropdown to select a row
            selected_row_index = st.selectbox("Select Row", range(len(filtered_price_range_df)), format_func=lambda i: f"Row {i}")
            selected_row = filtered_price_range_df.iloc[selected_row_index]

            # Create the scatter plot for the selected row
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(selected_row['MRP'], selected_row['DR1'], color='red', label='MRP Data Points')
            ax.scatter(selected_row['SP'], selected_row['DR2'], color='green', label='SP Data Points')
            ax.scatter(selected_row['MSP'], selected_row['DR3'], color='blue', label='MSP Data Points')

            # Perform polynomial regression to get the trendline
            z = np.polyfit(list(selected_row[['MRP', 'SP', 'MSP']]), list(selected_row[['DR1', 'DR2', 'DR3']]), 2)
            p = np.poly1d(z)

            # Generate x values for the trendline
            x_trendline = np.linspace(selected_row['MRP'].min(), selected_row['MSP'].max(), 100)

            # Plot the trendline
            ax.plot(x_trendline, p(x_trendline), linestyle='dashed', color='black', label='Trendline')

            # Add labels and title
            ax.set_xlabel('Price (MRP, SP, MSP)')
            ax.set_ylabel('DRs (DR1, DR2, DR3)')
            ax.set_title('Scatter Plot of MRP, SP, MSP vs. DR1, DR2, DR3')

            # Add legend
            ax.legend()
            
            st.write(f"Selected Row Dataset Values:")
            st.write(f"MRP: {selected_row['MRP']}, SP: {selected_row['SP']}, MSP: {selected_row['MSP']}, Base Price: {selected_row['Base Price']}")
            st.write(f"Demand Ratio(MRP): {selected_row['DR1']}, Demand Ratio(SP): {selected_row['DR2']}, Demand Ratio(MSP): {selected_row['DR3']}, Order Probablity: {selected_row['Probable Index for Bp']}")

            # Integrate the polynomial function to find the area under the curve
            area_under_curve = abs(np.trapz(p(x_trendline), x=x_trendline)) * 100
            st.write(f'Area under the curve: {area_under_curve}')
            turn_75 = abs(0.75 * area_under_curve) 
            turn_90 = abs(0.90 * area_under_curve) 
            st.write(f'75% of Turnover: {turn_75}')
            st.write(f'90% of Turnover: {turn_90}')
            # Average Turn Over
            x2 = selected_row['MRP']
            x1 = selected_row['MSP']
            avg_new = (area_under_curve) / (x2-x1) 
            # ---------------------------------------
            
            st.write(f'Average Turnover:{avg_new}')
             # Convert value to float
            value = float(value)

            # Calculate the result using the polynomial function p
            result = abs(p(value)*value)

            # Display the result on the screen
            st.write(f'Totat TurnOver at Give Price: {result}')
            
            # Display mean, median, and mode of the selected price range
            if not filtered_price_range_df.empty:
                
                selected_row_mean_price = filtered_price_range_df['MRP'].mean()
                selected_row_median_price = filtered_price_range_df['MRP'].median()
                selected_row_mode_price = filtered_price_range_df['MRP'].mode().iloc[0]  # In case there are multiple modes
            
            st.write(f"Median Price: {selected_row_median_price}")
            
            st.write(f"Mean Price: {selected_row_mean_price}")
            
            st.write(f"Mode Price: {selected_row_mode_price}")
            # Show the plot using Streamlit
            st.pyplot(fig)    

        else:
            st.warning("No data available for the selected price range.")
    else:
        st.warning("No data available for the selected date.")


elif selected_tab == 'History/Forecast':
    # History/Forecast Tab Content

    # Algorithm selection
    algorithm = st.selectbox("Select Algorithm", ["Linear Regression", "Extra Trees", "Decision Trees", "Gradient Boosting","HistGBM", "k-Nearest Neighbour", "XgBoost", "Random Forest Regressor"])

    if algorithm == "Linear Regression":
        lr_model_mrp = LinearRegression()
        lr_model_sp = LinearRegression()
        lr_model_msp = LinearRegression()

        # Train the models
        lr_model_mrp.fit(X_train, y_mrp_train)
        lr_model_sp.fit(X_train, y_sp_train)
        lr_model_msp.fit(X_train, y_msp_train)

        # Make predictions on the entire dataset for plotting
        df['MRP_Pred'] = lr_model_mrp.predict(X)
        df['SP_Pred'] = lr_model_sp.predict(X)
        df['MSP_Pred'] = lr_model_msp.predict(X)

    elif algorithm == "Extra Trees":
        et_regressor_mrp = ExtraTreesRegressor(n_estimators=100, random_state=42)
        et_regressor_sp = ExtraTreesRegressor(n_estimators=100, random_state=42)
        et_regressor_msp = ExtraTreesRegressor(n_estimators=100, random_state=42)

        # Train the models
        et_regressor_mrp.fit(X_train, y_mrp_train)
        et_regressor_sp.fit(X_train, y_sp_train)
        et_regressor_msp.fit(X_train, y_msp_train)

        # Make predictions on the entire dataset for plotting
        df['MRP_Pred'] = et_regressor_mrp.predict(X)
        df['SP_Pred'] = et_regressor_sp.predict(X)
        df['MSP_Pred'] = et_regressor_msp.predict(X)

    elif algorithm == "Decision Trees":
        dt_regressor_mrp = DecisionTreeRegressor(random_state=42)
        dt_regressor_sp = DecisionTreeRegressor(random_state=42)
        dt_regressor_msp = DecisionTreeRegressor(random_state=42)

        # Train the models
        dt_regressor_mrp.fit(X_train, y_mrp_train)
        dt_regressor_sp.fit(X_train, y_sp_train)
        dt_regressor_msp.fit(X_train, y_msp_train)

        # Make predictions on the entire dataset
        df['MRP_Pred'] = dt_regressor_mrp.predict(X)
        df['SP_Pred'] = dt_regressor_sp.predict(X)
        df['MSP_Pred'] = dt_regressor_msp.predict(X)

    elif algorithm == "Gradient Boosting":
        gb_regressor_mrp = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_regressor_sp = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_regressor_msp = GradientBoostingRegressor(n_estimators=100, random_state=42)

        # Train the models
        gb_regressor_mrp.fit(X_train, y_mrp_train)
        gb_regressor_sp.fit(X_train, y_sp_train)
        gb_regressor_msp.fit(X_train, y_msp_train)

        # Make predictions on the test set
        df['MRP_Pred'] = gb_regressor_mrp.predict(X)
        df['SP_Pred']= gb_regressor_sp.predict(X)
        df['MSP_Pred'] = gb_regressor_msp.predict(X)

    elif algorithm == "HistGBM":
        # Create HistGradientBoosting regressors for each target variable
        histgb_regressor_mrp = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        histgb_regressor_sp = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        histgb_regressor_msp = HistGradientBoostingRegressor(max_iter=100, random_state=42)

        # Train the models
        histgb_regressor_mrp.fit(X_train, y_mrp_train)
        histgb_regressor_sp.fit(X_train, y_sp_train)
        histgb_regressor_msp.fit(X_train, y_msp_train)

        # Make predictions on the test set
        df['MRP_Pred'] = histgb_regressor_mrp.predict(X)
        df['SP_Pred'] = histgb_regressor_sp.predict(X)
        df['MSP_Pred'] = histgb_regressor_msp.predict(X)
        
    elif algorithm == "k-Nearest Neighbour":
        # Create HistGradientBoosting regressors for each target variable
        knn_regressor_mrp = KNeighborsRegressor(n_neighbors=5)
        knn_regressor_sp = KNeighborsRegressor(n_neighbors=5)
        knn_regressor_msp = KNeighborsRegressor(n_neighbors=5)

        # Train the models
        knn_regressor_mrp.fit(X_train, y_mrp_train)
        knn_regressor_sp.fit(X_train, y_sp_train)
        knn_regressor_msp.fit(X_train, y_msp_train)

        # Make predictions on the test set
        df['MRP_Pred'] = knn_regressor_mrp.predict(X)
        df['SP_Pred'] = knn_regressor_sp.predict(X)
        df['MSP_Pred'] = knn_regressor_msp.predict(X)
        
    elif algorithm == "XgBoost":
        # Create HistGradientBoosting regressors for each target variable
        xgb_regressor_mrp = XGBRegressor(n_estimators=100, random_state=42)
        xgb_regressor_sp = XGBRegressor(n_estimators=100, random_state=42)
        xgb_regressor_msp = XGBRegressor(n_estimators=100, random_state=42)

        # Train the models
        xgb_regressor_mrp.fit(X_train, y_mrp_train)
        xgb_regressor_sp.fit(X_train, y_sp_train)
        xgb_regressor_msp.fit(X_train, y_msp_train)

        # Make predictions on the test set
        df['MRP_Pred'] = xgb_regressor_mrp.predict(X)
        df['SP_Pred'] = xgb_regressor_sp.predict(X)
        df['MSP_Pred'] = xgb_regressor_msp.predict(X)

    elif algorithm == "Random Forest Regressor":
        # Create HistGradientBoosting regressors for each target variable
        rf_regressor_mrp = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor_sp = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor_msp = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the models
        rf_regressor_mrp.fit(X_train, y_mrp_train)
        rf_regressor_sp.fit(X_train, y_sp_train)
        rf_regressor_msp.fit(X_train, y_msp_train)

        # Make predictions on the test set
        df['MRP_Pred'] = rf_regressor_mrp.predict(X)
        df['SP_Pred'] = rf_regressor_sp.predict(X)
        df['MSP_Pred'] = rf_regressor_msp.predict(X)

    # Repeat the same structure for other algorithms

    # Show Result button
    if st.button("Show Result"):
        # Downsample by aggregating into monthly averages
        df_resampled = df.resample('M', on='Date').mean()

        # Create the time series plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plotting the actual data
        ax.plot(df_resampled['MRP'], label='MRP (Actual)', color='red')
        ax.plot(df_resampled['SP'], label='SP (Actual)', color='green')
        ax.plot(df_resampled['MSP'], label='MSP (Actual)', color='blue')

        # Plotting the predictions
        ax.plot(df_resampled['MRP_Pred'], label='MRP Prediction', linestyle='dashed', color='orange')
        ax.plot(df_resampled['SP_Pred'], label='SP Prediction', linestyle='dashed', color='purple')
        ax.plot(df_resampled['MSP_Pred'], label='MSP Prediction', linestyle='dashed', color='brown')

        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'Plotting of MRP, SP, and MSP with {algorithm} Predictions')

        # Rotate x-axis labels for better readability
        ax.tick_params(rotation=45)

        # Add legend
        ax.legend()

        st.pyplot(fig)
