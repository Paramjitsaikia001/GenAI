from langchain_text_splitters import CharacterTextSplitter

text="""
🔹 DevTrack
DevTrack is a developer-focused learning and productivity platform that provides structured roadmaps, progress tracking, and AI-powered guidance. It helps developers stay organized, explore technologies, and grow their skills with a clean and efficient user experience.
Live Demo: https://trackofdev.vercel.app/ GitHub Repo: https://github.com/Paramjitsaikia001/trackofdev

🔹 Backing-Platform
A secure all-in-one banking web application built with Node.js, Express, MongoDB, and Firebase that enables users to manage accounts, transfer funds, and track transactions. Features include authentication, real-time balance updates, and a user-friendly dashboard. Live Demo: https://banking-platform-three.vercel.app
GitHub Repo: https://github.com/Paramjitsaikia001/banking-platform

🔹 CoworkAssam
A collaborative platform designed to help teams in Assam manage projects, communication, and tasks efficiently. Built with React, TailwindCSS, Node.js, Express, and MongoDB, it supports real-time collaboration and scalability. Live Demo: https://coworkassam.vercel.app/ GitHub Repo: https://github.com/sum1t-here/coworkassam

🔹 ExamEase
A web platform that helps teachers create MCQ/MSQ question papers with customizable templates, automated evaluation, and secure student access. Features include timed exams, unique access codes, and instant result delivery to teachers via email. Live Demo: https://paramjitsaikia001.github.io/ExamEase/ GitHub Repo: https://github.com/Paramjitsaikia001/ExamEase


"""

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=" "
)

result=splitter.split_text(text)

print(result)