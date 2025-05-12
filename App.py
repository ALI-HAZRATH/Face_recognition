import streamlit as st
import cv2
import numpy as np
import face_recognition
import pandas as pd
import datetime
from Image_Encoding import known_face_encodings, known_face_names

# Streamlit page setup
st.set_page_config(page_title="🎓 Face Recognition Attendance", page_icon="📸", layout="wide")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Attendance file
excel_file = "attendance.xlsx"
try:
    df = pd.read_excel(excel_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Name", "Date", "Time"])

# Streamlit Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["📸 Face Recognition", "📊 College Data (Attendance Tracking)"])

# ============================== #
# ✅ FACE RECOGNITION PAGE
# ============================== #
if page == "📸 Face Recognition":
    # Stylish header with color
    st.markdown("<h1 style='text-align: center; color: #008080;'>📸 Face Recognition Attendance System</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #008080;'>", unsafe_allow_html=True)

    # Intro text with a clean and friendly tone
    st.markdown("""
    Welcome to the **Face Recognition Attendance System**!  
    This system automatically records student attendance using **face recognition**.  
    Simply click on "Start Recognition" to begin, and the system will mark attendance in real-time.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Function to mark attendance
    def mark_attendance(name):
        global df
        now = datetime.datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        try:
            df = pd.read_excel(excel_file)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Name", "Date", "Time"])

        if "Date" not in df.columns:
            df["Date"] = ""
        if "Time" not in df.columns:
            df["Time"] = ""

        # Check if attendance for this name has already been marked today
        if not ((df["Name"] == name) & (df["Date"] == current_date)).any():
            new_entry = pd.DataFrame([{"Name": name, "Date": current_date, "Time": current_time}])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_excel(excel_file, index=False)
            st.success(f"✅ Attendance marked for {name} on {current_date} at {current_time}")
        else:
            st.warning(f"⚠ {name} has already marked attendance today.")

    # Buttons to Start and Stop Recognition
    run = st.button("▶ Start Recognition")
    stop = st.button("⏹ Stop Recognition")

    if run:
        stframe = st.empty()
        stop_signal = False

        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error("❌ Error: Couldn't access the webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches and matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    mark_attendance(name)

                # Draw rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Display frame in Streamlit
            stframe.image(frame, channels="BGR")

            # Check if "Stop Recognition" button is pressed
            if stop:
                stop_signal = True

            if stop_signal:
                break

        video_capture.release()
        cv2.destroyAllWindows()

# ==============================
# ✅ COLLEGE DATA PAGE (WITH DASHBOARD)
# ==============================
elif page == "📊 College Data (Attendance Tracking)":
    st.markdown("<h1 style='text-align: center; color: #1E90FF;'>📊 College Attendance Tracking</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #1E90FF;'>", unsafe_allow_html=True)

    # Load the attendance data
    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        st.error("❌ No attendance data found.")
        st.stop()

    # Filter options
    date_filter = st.date_input("📅 Select Date", datetime.date.today())
    date_filter = date_filter.strftime("%Y-%m-%d")  # Convert date to string

    # Get the list of students present on the selected date
    present_students = df[df["Date"] == date_filter]["Name"].tolist()

    # Compare with known students to find absentees
    known_students = set(known_face_names)
    present_students = set(present_students)
    absentees = list(known_students - present_students)

    # ==============================
    # 📈 Live Attendance Dashboard
    # ==============================
    st.markdown("## 📈 Live Attendance Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="🎯 Total Students", value=len(known_face_names))

    with col2:
        st.metric(label="👨‍🎓 Students Present", value=len(present_students))

    with col3:
        st.metric(label="❌ Students Absent", value=len(absentees))

    with col4:
        st.metric(label="📅 Date", value=datetime.date.today().strftime("%d %b, %Y"))

    st.markdown("<hr style='border: 1px solid #1E90FF;'>", unsafe_allow_html=True)

    # Celebration if 100% Attendance
    if len(absentees) == 0:
        st.balloons()

    # Show attendance data
    st.markdown("### 📋 Attendance Records")
    st.dataframe(df[df["Date"] == date_filter], use_container_width=True)

    # Show absentees
    st.markdown("### ❌ Absentees List")
    if absentees:
        st.warning(f"⚠ The following students were absent on {date_filter}:")
        for student in absentees:
            st.info(f"{student}")
    else:
        st.success(f"✅ No absentees on {date_filter}!")

    # Download Attendance Data
    st.markdown("### 📥 Download Attendance Data")
    csv = df[df["Date"] == date_filter].to_csv(index=False).encode("utf-8")
    st.download_button(label="📂 Download CSV", data=csv, file_name="attendance_data.csv", mime="text/csv")


