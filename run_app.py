import streamlit.web.cli as stcli
import sys

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "CV_IM.py", "--server.headless=true"]
    stcli.main()
