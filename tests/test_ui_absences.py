"""Tkinter regression: 20 resolved simulated faces must show five absences, not zero.

The window stays hidden; model/generator fixtures isolate the UI's state changes.
Real-model absence scenarios are tested separately in src.absence_validation.
"""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import pandas as pd
from src.common import roster, write_json
from src import ui


class AbsenceUITests(unittest.TestCase):
    def test_twenty_faces_automatically_show_five_absences(self):
        students = roster()
        self.assertEqual(len(students),25)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = 'ui_absence_test'
            path = root/'sessions'/session

            def fake_generate(count,seed,split):
                self.assertEqual(count,20)
                path.mkdir(parents=True)
                cv2.imwrite(str(path/'classroom.png'),np.zeros((400,600,3),np.uint8))
                write_json(path/'simulation.json',{'kind':'synthetic_classroom','session_id':session,
                    'registered_count':25,'generated_count':20,'expected_absent_count':5})
                return session

            def fake_recognize(current):
                self.assertEqual(current,session)
                faces = [{'name':s['Name'],'distance':.1,'candidates':[s['Name']]} for s in students[:20]]
                result = {'session_id':session,'timestamp':'2026-10-29T10:00:00+01:00',
                          'faces':faces,'detected_count':20,'recognized_count':20,'errors':[]}
                write_json(path/'predictions.json',result)
                return result

            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)

            def drive(window):
                window.withdraw()
                def wait_for(condition):
                    deadline = time.monotonic()+8
                    while not condition():
                        window.update()
                        if errors.called: self.fail(str(errors.call_args))
                        if time.monotonic()>deadline: self.fail('UI did not finish updating attendance')
                        time.sleep(.01)
                try:
                    widgets = list(descendants(window))
                    spin = next(w for w in widgets if isinstance(w,ttk.Spinbox))
                    buttons = {str(w.cget('text')):w for w in widgets if isinstance(w,ttk.Button)}
                    notebook = next(w for w in widgets if isinstance(w,ttk.Notebook))
                    spin.set('20'); buttons['Generate'].invoke()
                    wait_for(lambda:any('Absences prévues : 5' in str(window.getvar(w.cget('textvariable')))
                        for w in widgets if isinstance(w,ttk.Label) and w.cget('textvariable')))
                    buttons['Recognize'].invoke()
                    wait_for(lambda:any(notebook.tab(tab,'text')=='Absents confirmés (5)' for tab in notebook.tabs()))
                    exported = pd.read_csv(root/'attendance_reports'/f'{session}_absents.csv')
                    self.assertEqual(set(exported.Name),{s['Name'] for s in students[20:]})
                    self.assertTrue(exported.Student_ID.is_unique)
                finally:
                    window.tk.call(window.protocol('WM_DELETE_WINDOW'))

            with patch('src.ui.ROOT',root), patch('src.session_runner.ROOT',root), \
                 patch('src.attendance_manager.ROOT',root), \
                 patch('src.ui.generate',side_effect=fake_generate), \
                 patch('src.ui.recognize',side_effect=fake_recognize), \
                 patch('src.ui.messagebox.showerror') as errors, \
                 patch.object(tk.Tk,'mainloop',drive):
                ui.launch()


if __name__=='__main__': unittest.main()
