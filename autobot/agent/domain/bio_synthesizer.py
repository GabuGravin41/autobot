"""
BioInformatics & Gene Annotation Synthesizer — Generates BioPython scripts for NCBI BLAST searches, accession lookups, and sequence parsing.
"""
import logging

logger = logging.getLogger(__name__)


class BioInformaticsSynthesizer:
    """
    Synthesizes BioPython scripts for remote NCBI BLAST searches,
    fasta sequence fetching, and alignment parsing.
    """

    @staticmethod
    def generate_blast_script(sequence_or_fasta: str, program: str = "blastn", database: str = "nr", output_csv: str = "blast_results.csv") -> str:
        """
        Generates a standalone Python script to execute an NCBI BLAST query via BioPython
        and export alignments to CSV.
        """
        # repr(), not raw f-string interpolation into a quoted literal. The
        # previous version did `sequence = """{sequence_or_fasta}"""` — any
        # input containing `"""` (or, for program/database/output_csv,
        # a bare `"`) breaks out of the string literal in the GENERATED
        # script, turning arbitrary text into arbitrary Python that then
        # gets written to disk and potentially executed via run_command.
        # repr() produces a properly quote-and-backslash-escaped literal no
        # matter what the input contains.
        sequence_lit = repr(sequence_or_fasta)
        program_lit = repr(program)
        database_lit = repr(database)
        output_csv_lit = repr(output_csv)
        return f'''# Auto-generated BioInformatics BLAST Script
import sys
import csv

try:
    from Bio.Blast import NCBIWWW, NCBIXML
except ImportError:
    print("⚠️ BioPython not found. Install with 'pip install biopython'.")
    sys.exit(1)

sequence = {sequence_lit}
program = {program_lit}
database = {database_lit}
output_csv = {output_csv_lit}

print(f"🧬 Running NCBI BLAST ({{program}} on {{database}})...")
result_handle = NCBIWWW.qblast(program, database, sequence)

print("🔍 Parsing BLAST XML results...")
blast_records = NCBIXML.parse(result_handle)

results = []
for record in blast_records:
    for alignment in record.alignments[:5]:
        for hsp in alignment.hsps:
            results.append({{
                "title": alignment.title,
                "length": alignment.length,
                "e_value": hsp.expect,
                "score": hsp.score,
                "identities": hsp.identities,
            }})

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "length", "e_value", "score", "identities"])
    writer.writeheader()
    writer.writerows(results)

print(f"✅ Saved {{len(results)}} BLAST alignments to {{output_csv}}")
'''
