import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if "first_time" not in st.session_state:
    st.session_state["first_time"] = True

st.set_page_config(page_title='HELOC Risk Evaluation', page_icon='💰', layout='wide',
                   initial_sidebar_state='auto', menu_items={
                        'Get Help': None,
                        'Report a bug': None,
                        'About': '''
                        This interface was made by **Team 5** and its purpose is to help bank sale stuff to evaluate the customers' credit performance. 
                        
                        '''
     })


#初始数值的设定
ExternalRiskEstimate         =         72
NumSatisfactoryTrades         =        21
MaxDelq2PublicRecLast12M       =        6
MSinceMostRecentInqexcl7days    =       3
NumInqLast6M                     =      2
NumInqLast6Mexcl7days             =     1
NumRevolvingTradesWBalance         =    4
NumBank2NatlTradesWHighUtilization  =   1

def load_model(model_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    return model

# 加载 scaler 对象
with open("scaler_data.p", "rb") as f:
    scaler = pickle.load(f)

# 加载 pca 对象
with open("pca.p", "rb") as f:
    pca = pickle.load(f)

def predict_credit_risk(model, features):
    # 使用模型进行预测
    result = model.predict([features])
    return result[0]

def get_risk_category(result):
    if result == 1:
        return "GOOD"
    else:
        return "BAD"


def input_to_radarchart(scaled_df):
    def plot_radar_chart(df):
        categories = df.columns
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]      
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for i, row in df.iterrows():
            row[-2:] = row[-2:] * -1
            values = row.values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid',label="Client Risk Performance")
            
            ax.fill(angles, values, alpha=0.25)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        categories = ["Credit Inquiries \nand \nCredit Activity \nLevel",
                            "Credit Behavior \nand Credit Utilization", 
                            "Credit Risk \nand  \nPublic Record \nDelinquency", 
                            "Recent \nCredit Inquiries"]
        ax.set_xticklabels(categories)
        ax.tick_params(axis='x', which='major', pad=30)
        ax.set_title("Client Risk Profile", fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1.1))
        st.pyplot(fig)
    transformed_data = pd.DataFrame(pca.transform(scaled_df))
    plot_radar_chart(transformed_data)

advice = {
    '0': {
        'good': "Great performance! You have a low number of credit inquiries and a moderate credit activity level. Continue maintaining good credit management habits.",
        'poor': "Needs improvement: Be mindful of controlling the number of credit inquiries, as too many inquiries can negatively impact your credit score. Also, maintain an appropriate credit activity level to uphold a healthy credit standing."
    },
    '1': {
        'good': "Great performance! Your credit behavior is good, and your credit utilization is low. Continue maintaining good credit usage habits.",
        'poor': "Needs improvement: Try to improve your credit behavior, such as making timely payments, and work on reducing your credit utilization. This will help to improve your credit score."
    },
    '2': {
        'good': "Great performance! Your credit risk is low, and you have no public record delinquency. Continue maintaining a good credit record.",
        'poor': "Needs improvement: Strive to lower your credit risk and avoid public record delinquency. Improving these conditions will have a positive impact on your credit score."
    },
    '3': {
        'good': "Great performance! You have a low number of recent credit inquiries. Maintaining this situation will help uphold a good credit score.",
        'poor': "Needs improvement: Be mindful of controlling the number of recent credit inquiries, as too many inquiries can negatively impact your credit score. Be cautious when applying for new credit lines."
    }
}


def main():

    #st.title("Home Equity Line of Credit (HELOC) Risk Evaluation")
    #st.title("Home Equity Line of Credit (HELOC) Risk Evaluation")



    st.title("HELOC Risk Evaluation System")
    st.subheader("Interactive Interface for Sales Representatives")
    st.caption('Made by LU')

    st.markdown('''
            <p style="font-size:18px;">Welcome to the Home Equity Line of Credit (HELOC) risk evaluation system. This system is designed to help sales representatives in a bank or credit card company evaluate HELOC applications.</p>

            <p style="font-size:18px;">To use the system, simply fill out the form on the left-hand side with the applicant's information, and click the <span style="color:green;font-weight:bold;">Start Evaluation</span> button to get a risk assessment. The results will appear on the right-hand side of the screen.</p>

            <p style="font-size:18px; color:red;">Please note that the data you enter is not saved by the system!</p>
            ''', unsafe_allow_html=True)



    def provide_advice(principal_components_df):
        advice_output = {}

        for component in map(str, principal_components_df.columns):
            value = principal_components_df.at[0, component]

            if component in ['0', '1']:
                if value > 0:
                    advice_output[component] = advice[component]['good']
                else:
                    advice_output[component] = advice[component]['poor']
            else:
                if value < 0:
                    advice_output[component] = advice[component]['good']
                else:
                    advice_output[component] = advice[component]['poor']

        return advice_output


   # 在此处替换模型文件的路径
    model_file = "lda_model.p"
    model = load_model(model_file)

    # 用于收集用户输入的字典
    user_input = {}

    # 创建输入框
    with st.sidebar.form(key="input_form"):
        st.header('Please enter the following information about the applicant:')
        st.subheader('Credit Risk Factors')


        user_input["ExternalRiskEstimate"] = st.slider("What is the applicant's External Risk Estimate (comprehensive version)?", value=72, min_value=-10, step=1)

        user_input["NumSatisfactoryTrades"] = st.number_input("How many satisfactory trades does the applicant have?", value=21, min_value=-10, step=1)

        user_input["MaxDelq2PublicRecLast12M"] = st.number_input("What is the maximum delinquency or public record the applicant has had in the last 12 months? ", value=6, min_value=-10, step=1)

        user_input["MSinceMostRecentInqexcl7days"] = st.number_input("How many months has it been since the applicant's most recent inquiry, excluding the last 7 days? ", value=3, min_value=-10, step=1)

        user_input["NumInqLast6M"] = st.number_input("How many inquiries has the applicant had in the last 6 months? ", value=2, min_value=-10, step=1)

        user_input["NumInqLast6Mexcl7days"] = st.number_input("How many inquiries has the applicant had in the last 6 months, excluding the last 7 days? ", value=1, min_value=-10, step=1)

        user_input["NumRevolvingTradesWBalance"] = st.number_input("How many revolving trades with balance does the applicant have? ", value=4, min_value=-10, step=1)

        user_input["NumBank2NatlTradesWHighUtilization"] = st.number_input("How many bank/national trades does the applicant have with high utilization ratio? ", value=1, min_value=-10, step=1)

        st.header('')
        submit_button = st.form_submit_button("Start Evaluation")
        st.header('')

    if submit_button:


        # 在此处定义 input_data
        input_data = np.array([
            user_input["ExternalRiskEstimate"],
            user_input["NumSatisfactoryTrades"],
            user_input["MaxDelq2PublicRecLast12M"],
            user_input["MSinceMostRecentInqexcl7days"],
            user_input["NumInqLast6M"],
            user_input["NumInqLast6Mexcl7days"],
            user_input["NumRevolvingTradesWBalance"],
            user_input["NumBank2NatlTradesWHighUtilization"]
        ])

        # 调用预测函数并显示结果
        result = predict_credit_risk(model, list(user_input.values()))
        risk_category = get_risk_category(result)

        st.header("Credit Risk Evaluation Result")
        st.write(f"Based on the indicators you input, the predicted credit risk evaluation is: {risk_category}")

        col1, col2 = st.columns([3, 2])

        with col2:
            x1 = [0, 6, 0]
            x2 = [0, 3, 0]

            y = ['0', '1', '2']

            f, ax = plt.subplots(figsize=(5,2))

            p1 = sns.barplot(x=x1, y=y, color='#3EC300')
            p1.set(xticklabels=[], yticklabels=[])
            p1.tick_params(bottom=False, left=False)
            p2 = sns.barplot(x=x2, y=y, color='#FF331F')
            p2.set(xticklabels=[], yticklabels=[])
            p2.tick_params(bottom=False, left=False)
 

            plt.text(1.3, 1.05, "BAD", horizontalalignment='left', size='medium', color='white', weight='semibold')
            plt.text(4.1, 1.05, "GOOD", horizontalalignment='left', size='medium', color='white', weight='semibold')
     
            ax.set(xlim=(0, 6))
            sns.despine(left=True, bottom=True)

            figure = st.pyplot(f)


            placeholder = st.empty()

            if risk_category == 'GOOD':
                st.balloons()
                t1 = plt.Polygon([[4.45, 0.5], [4.95, 0], [3.95, 0]], color='black')
                placeholder.markdown('The credit score is **GOOD**! Congratulations on having a strong credit standing!')
                
            else:
                t1 = plt.Polygon([[1.5, 0.5], [2.0, 0], [1.0, 0]], color='black')
                placeholder.markdown('The credit score is POOR. We recommend taking steps to improve your credit standing as soon as possible.')
                
    

            plt.gca().add_patch(t1)
            figure.pyplot(f)
            prob_fig, ax = plt.subplots()

        with col1:




            with st.expander('Click to see the detailed analysis and advice'):
            
             # 对数据进行标准化
                scaled_data = scaler.transform(input_data.reshape(1, -1))

            # 对标准化后的数据进行PCA处理
                pca_data = pca.transform(scaled_data)

            # 使用标准化后的数据绘制雷达图
                input_to_radarchart(scaled_data)
           
            # 将 PCA 数据转换为 DataFrame
                pca_data_df = pd.DataFrame(pca_data, columns=[str(i) for i in range(pca_data.shape[1])])
            
            # Call provide_advice function
                advice_results = provide_advice(pca_data_df)
                for component, component_advice in advice_results.items():
                    st.write(f"Advice {component}: {component_advice}")
 




dialog_key = "welcome_dialog"

if st.session_state["first_time"]:
    welcome_message_1 = st.empty()
    welcome_message_2 = st.empty()
    welcome_message_1.markdown("<div style='text-align:center;font-size:24px;font-weight:bold;padding-right:5ch;margin-top:180px;'>Welcome to the HELOC risk evaluation system!</div>", unsafe_allow_html=True)
    welcome_message_2.markdown("<div style='text-align:center;font-size:24px;font-weight:bold;padding-right:6ch;margin-top:10px;'>Have you used this system before?</div>", unsafe_allow_html=True)

    col_space1, col_space2, col1, col2, col_space1, col_space1 = st.columns([1, 1, 1, 1, 1, 1])
    button1 = col1.empty()
    button2 = col2.empty()

    button1_clicked = button1.button("First time User", key="button1")
    button2_clicked = button2.button("Returning User", key="button2")

    if button1_clicked:
        st.session_state["first_time"] = False
        welcome_message_1.empty()
        welcome_message_2.empty()
        button1.empty()
        button2.empty()
        st.markdown("<div style='text-align:center;font-size:32px;font-weight:bold;padding-right:5ch;margin-top:-50px;'>Welcome to this system!</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;font-size:24px;font-weight:bold;padding-right:5ch;margin-top:-20px;'>Instructions for Use and Input Variable Explanation:</div>", unsafe_allow_html=True)
        
        st.markdown("This is a Python program built using Streamlit that allows sales representatives from a bank or credit card company to evaluate Home Equity Line of Credit (HELOC) applications. The purpose of this interface is to predict the credit risk of an applicant based on their input data.")
        st.markdown("The following input variables are used in this program to assess the credit risk of an applicant:")
        st.markdown("* ExternalRiskEstimate: the comprehensive version of the applicant's external risk estimate.")
        st.markdown("* NumSatisfactoryTrades: the number of satisfactory trades that the applicant has.")
        st.markdown("* MaxDelq2PublicRecLast12M: the maximum delinquency or public record that the applicant has had in the last 12 months.")
        st.markdown("* MSinceMostRecentInqexcl7days: the number of months since the applicant's most recent inquiry, excluding the last 7 days.")
        st.markdown("* NumInqLast6M: the number of inquiries that the applicant has had in the last 6 months.")
        st.markdown("* NumInqLast6Mexcl7days: the number of inquiries that the applicant has had in the last 6 months, excluding the last 7 days.")
        st.markdown("* NumRevolvingTradesWBalance: the number of revolving trades that the applicant has with balance.")
        st.markdown("* NumBank2NatlTradesWHighUtilization: the number of bank/national trades that the applicant has with a high utilization ratio.")

        st.markdown("Once the user inputs the values for these variables, the program applies a machine learning model (LDA) to predict the applicant's credit risk evaluation. The result is displayed on the right side of the screen as either 'GOOD' or 'BAD'. Additionally, the program generates advice for the user based on the applicant's input data. The advice is displayed in the collapsible 'Click to see the detailed analysis and advice' section. Finally, the program also generates a radar chart to visually represent the applicant's credit risk profile.")
       
        col_space1, col_space2, col2, col3, col_space1, col_space1, col_space1 = st.columns([1, 1, 1, 1, 1, 1, 1])

        if col3.button("Enter Homepage", key="button3"):
            welcome_message_1.empty()
            welcome_message_2.empty()
            button1.empty()
            button2.empty()
            main()

    elif button2_clicked:
        st.session_state["first_time"] = False
        welcome_message_1.empty()
        welcome_message_2.empty()  # 清除 "请选择您是否是第一次使用本应用程序：" 文字
        button1.empty()  # 清除 "第一次使用？" 按钮
        button2.empty()  # 清除 "非第一次使用？" 按钮
        main()

else:
    main()










# 调用 main() 函数
#main()




