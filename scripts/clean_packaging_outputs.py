import re
from pathlib import Path
from click.testing import CliRunner
from audiotrove.cli.main import cli

# Clean existing curate demo output
p = Path('packaging/curate_demo_output.txt')
if p.exists():
    txt = p.read_text(encoding='utf-8')
    ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    clean = ansi_re.sub('', txt).replace('\\', '/')
    Path('packaging/curate_demo_output_clean.txt').write_text(clean, encoding='utf-8')
    print('wrote packaging/curate_demo_output_clean.txt')

# Generate inspect output
runner = CliRunner()
res = runner.invoke(cli, ['inspect', 'tests/fixtures', '--extensions', 'wav,flac', '--limit', '5'])
inspect_out = res.output
ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
inspect_clean = ansi_re.sub('', inspect_out).replace('\\', '/')
Path('packaging/inspect_demo_output_clean.txt').write_text(inspect_clean, encoding='utf-8')
print('wrote packaging/inspect_demo_output_clean.txt')
