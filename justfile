set shell := ["powershell.exe", "-c"]

run:
    .venv\Scripts\python run.py

install:
    if ( Test-Path -Path .\.venv -PathType Container ) { exit 0 } else { py -m venv .venv; .venv\Scripts\pip install -r requirements.txt }

clean:
    Remove-Item -Path ".venv" -Recurse -Force -ErrorAction SilentlyContinue

re: clean install