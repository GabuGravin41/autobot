from __future__ import annotations
import time
from typing import Any
from .base import ActionSpec, BaseAdapter

class NCBIBlastWebAdapter(BaseAdapter):
    name = "ncbi_blast_web"
    description = "Run protein/nucleotide BLAST on NCBI: submit sequences, wait for job, read top hits."
    actions = {
        "open_home": ActionSpec("Open NCBI BLAST home page"),
        "open_protein_blast": ActionSpec("Open BLASTP (protein) page"),
        "submit_sequence": ActionSpec("Enter FASTA sequence and click BLAST"),
        "check_job_status": ActionSpec("Check if results are ready"),
        "get_top_hit": ActionSpec("Read the first hit accession and description"),
    }

    def do_open_home(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://blast.ncbi.nlm.nih.gov/Blast.cgi")
        return "Opened NCBI BLAST home."

    def do_open_protein_blast(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastp&PAGE_TYPE=BlastSearch&LINK_LOC=blasthome")
        if self._human_mode():
            time.sleep(5)
        return "Opened Protein BLAST."

    def do_submit_sequence(self, params: dict[str, Any]) -> str:
        sequence = str(params.get("sequence", "")).strip()
        if not sequence:
            raise ValueError("submit_sequence requires 'sequence'.")
        
        if self._human_mode():
            # In human mode, we rely on the focus and typing
            self.run_human_nav("submit_sequence", {"sequence": sequence})
            return "Sequence submitted (human mode)."
            
        # Devtools mode
        self.browser.start()
        # The main input area is usually a textarea with ID 'QUERY'
        self.browser.fill("#QUERY", sequence)
        time.sleep(1)
        # The BLAST button has class 'blastbutton' or similar
        self.browser.click(".blastbutton")
        return "Sequence submitted to NCBI BLAST."

    def do_check_job_status(self, _params: dict[str, Any]) -> str:
        self.browser.start()
        # NCBI shows a "Job title" and "Status: Searching..." or "Status: Done"
        # Often there's an element like #contentHeader status
        try:
            status_loc = self.browser.page.locator("#contentHeader .status").first
            if status_loc.is_visible(timeout=2000):
                return status_loc.inner_text().strip()
            
            # If we see the results table, it's done
            if self.browser.page.locator("#dscTable").first.is_visible(timeout=2000):
                return "Done"
        except Exception:
            pass
        return "Unknown (status element not found)"

    def do_get_top_hit(self, _params: dict[str, Any]) -> str:
        self.browser.start()
        try:
            # The hits table is usually #dscTable
            # First row [1] in tbody, usually has description and accession
            first_hit_desc = self.browser.page.locator("#dscTable tbody tr").first.locator("td").nth(2).inner_text()
            first_hit_acc = self.browser.page.locator("#dscTable tbody tr").first.locator("td").nth(3).inner_text()
            return f"Top Hit: {first_hit_acc} - {first_hit_desc}"
        except Exception as e:
            return f"Error reading hits: {e}"
