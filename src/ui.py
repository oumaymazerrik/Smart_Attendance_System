"""Small Windows desktop UI using Tkinter; inference runs off the UI thread."""
import concurrent.futures
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from .common import ROOT, roster, read_json
from .classroom_generator import generate
from .session_runner import recognize, compare_truth, session_path
from .attendance_manager import (attendance_table, export_attendance, attendance_groups,
                                 simulation_info, simulation_completion)


def launch():
    window = tk.Tk()
    window.title('Smart Attendance • Synthetic Classroom Lab')
    width, height = min(1180, window.winfo_screenwidth()-60), min(850, window.winfo_screenheight()-90)
    window.geometry(f'{width}x{height}')
    window.configure(bg='#edf1f5')
    style = ttk.Style(); style.theme_use('clam')
    style.configure('.', font=('Segoe UI',9), background='#edf1f5')
    style.configure('TButton', padding=(8,6))
    style.configure('Treeview', background='white', fieldbackground='white', rowheight=23)
    style.configure('Treeview.Heading', font=('Segoe UI',9,'bold'))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    session, photo, future = None, None, None
    ttk.Label(window, text='Smart Attendance', font=('Segoe UI',22,'bold')).pack(anchor='w', padx=24, pady=(16,0))
    ttk.Label(window, text='FaceNet512 • fictional student simulation • review uncertain matches').pack(anchor='w', padx=24,pady=(0,10))
    controls = ttk.Frame(window); controls.pack(fill='x',padx=24,pady=6)
    ttk.Label(controls,text='Students').pack(side='left')
    count = tk.IntVar(value=len(roster()))
    ttk.Spinbox(controls,from_=0,to=len(roster()),textvariable=count,width=4).pack(side='left',padx=6)
    ttk.Label(controls,text='Seed').pack(side='left')
    seed = tk.StringVar(value='42')
    ttk.Entry(controls,textvariable=seed,width=9).pack(side='left',padx=6)
    ttk.Label(controls,text='Portrait pool').pack(side='left')
    split = tk.StringVar(value='test')
    ttk.Combobox(controls,textvariable=split,values=['test','validation','train'],state='readonly',width=10).pack(side='left',padx=6)
    status = tk.StringVar(value='Generate a classroom session, then run recognition. Orange matches need inspection.')
    ttk.Label(window,textvariable=status,wraplength=1100).pack(anchor='w',padx=24,pady=5)
    simulation_counts = tk.StringVar(value=f'Inscrits : {len(roster())} | Générez une séance pour connaître les absences prévues.')
    ttk.Label(window,textvariable=simulation_counts,font=('Segoe UI',10,'bold')).pack(anchor='w',padx=24,pady=(0,3))
    image_label = ttk.Label(window,anchor='center',text='Generate a classroom session to begin.'); image_label.pack(fill='both',expand=True,padx=24,pady=4)
    image_label.configure(cursor='hand2')
    columns = ['Student_ID','Name','Date','Time','Status','Distance']
    notebook = ttk.Notebook(window)
    tables = {}
    for key,label in [('all','Tous'),('present','Présents'),('absent','Absents confirmés'),('review','À vérifier')]:
        frame = ttk.Frame(notebook)
        tree = ttk.Treeview(frame,columns=columns,show='headings',height=5)
        for c in columns: tree.heading(c,text=c); tree.column(c,width=130,anchor='center')
        scrollbar = ttk.Scrollbar(frame,orient='vertical',command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side='left',fill='both',expand=True); scrollbar.pack(side='right',fill='y')
        notebook.add(frame,text='Absents — non analysés' if key=='absent' else f'{label} (0)'); tables[key]=(frame,tree,label)
    notebook.pack(fill='x',padx=24,pady=(6,2))
    absence_note = tk.StringVar(value='Les absences sont confirmées après finalisation et vérification de la couverture.')
    ttk.Label(window,textvariable=absence_note,wraplength=1100).pack(anchor='w',padx=24,pady=(0,3))
    policy = ttk.Frame(window); policy.pack(fill='x',padx=24,pady=4)
    completed, coverage = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
    ttk.Checkbutton(policy,text='Séance terminée',variable=completed).pack(side='left')
    ttk.Checkbutton(policy,text='Couverture des visages vérifiée',variable=coverage).pack(side='left',padx=14)
    ttk.Label(window,text='Orange = Needs Review    Red = Unknown    Green = accepted identity. Distance is cosine distance, not probability.').pack(anchor='w',padx=24,pady=(3,12))

    def show_table(table):
        groups = attendance_groups(table)
        groups['all'] = table
        for key,(frame,tree,label) in tables.items():
            tree.delete(*tree.get_children())
            for values in groups[key].itertuples(index=False,name=None):
                tree.insert('', 'end', values=[f'{v:.4f}' if isinstance(v,float) and v == v else '' if v is None or v != v else v for v in values])
            notebook.tab(frame,text=f'{label} ({len(groups[key])})')
        if completed.get() and coverage.get():
            absence_note.set(f"Absents confirmés : {len(groups['absent'])} | À vérifier : {len(groups['review'])}. Les cas incertains ne sont pas déclarés absents.")
        else:
            absence_note.set(f"Absences non finalisées | Non reconnus : {len(groups['not_recognized'])}. Vérifiez la couverture des visages avant confirmation.")
            frame,_,label = tables['absent']
            notebook.tab(frame,text='Absents — à confirmer')

    def clear_tables():
        for key,(frame,tree,label) in tables.items():
            tree.delete(*tree.get_children())
            notebook.tab(frame,text='Absents — non analysés' if key=='absent' else f'{label} (0)')

    def refresh_attendance(*args):
        if not session: return
        if future is not None and not future.done(): return
        path = session_path(session)/'predictions.json'
        if path.exists():
            result = read_json(path)
            show_table(attendance_table(result['faces'],completed.get(),coverage.get() and not result['errors'],result['timestamp']))
    completed.trace_add('write',refresh_attendance)
    coverage.trace_add('write',refresh_attendance)

    def full_image(event=None):
        if not session: return
        directory = session_path(session)
        path = directory/'annotated.png'
        if not path.exists(): path = directory/'classroom.png'
        viewer = tk.Toplevel(window); viewer.title(f'Classroom detail • {session}')
        viewer.geometry(f'{width}x{height}')
        viewer.rowconfigure(0,weight=1); viewer.columnconfigure(0,weight=1)
        canvas = tk.Canvas(viewer,highlightthickness=0)
        vertical = ttk.Scrollbar(viewer,orient='vertical',command=canvas.yview)
        horizontal = ttk.Scrollbar(viewer,orient='horizontal',command=canvas.xview)
        canvas.configure(yscrollcommand=vertical.set,xscrollcommand=horizontal.set)
        canvas.grid(row=0,column=0,sticky='nsew'); vertical.grid(row=0,column=1,sticky='ns')
        horizontal.grid(row=1,column=0,sticky='ew')
        picture = ImageTk.PhotoImage(Image.open(path))
        canvas.create_image(0,0,image=picture,anchor='nw'); canvas.image = picture
        canvas.configure(scrollregion=canvas.bbox('all'))
    image_label.bind('<Button-1>',full_image)

    def display(current):
        nonlocal session, photo
        session = current
        directory = session_path(session)
        path = directory / 'annotated.png'
        if not path.exists(): path = directory / 'classroom.png'
        image = Image.open(path); image.thumbnail((width-70,max(120,height-500)))
        photo = ImageTk.PhotoImage(image); image_label.configure(image=photo)
        info = simulation_info(session)
        if info:
            simulation_counts.set(f"Inscrits : {info['registered_count']} | Dans la simulation : {info['generated_count']} | Absences prévues : {info['expected_absent_count']}")
        else:
            simulation_counts.set(f'Inscrits : {len(roster())} | Nombre de personnes attendues non renseigné : couverture à confirmer.')
        completed.set(False); coverage.set(False)
        if (directory/'attendance_policy.json').exists():
            saved = read_json(directory/'attendance_policy.json')
            completed.set(saved['completed']); coverage.set(saved['coverage_adequate'])
        if (directory/'predictions.json').exists():
            result = read_json(directory/'predictions.json')
            if not (directory/'attendance_policy.json').exists():
                auto = simulation_completion(result,info)
                completed.set(auto['completed']); coverage.set(auto['coverage_adequate'])
            show_table(attendance_table(result['faces'],completed.get(),coverage.get() and not result['errors'],result['timestamp']))
            summary = f" | {result['recognized_count']}/{result['detected_count']} accepted"
        else:
            clear_tables(); summary = ''
            absence_note.set('Lancez la reconnaissance pour établir la présence et les absences.')
        status.set(f'Session {session}{summary}')

    def submit(fn, callback):
        nonlocal future
        if future is not None and not future.done(): return
        status.set('Working…')
        future = pool.submit(fn)
        def poll():
            if not future.done(): window.after(100,poll); return
            try: callback(future.result())
            except Exception as exc: status.set(str(exc)); messagebox.showerror('Operation failed',str(exc))
        window.after(100,poll)

    def new():
        try:
            n, s, source_split = count.get(), int(seed.get()) if seed.get().strip() else None, split.get()
            submit(lambda: generate(n,s,source_split),display)
        except Exception as exc: messagebox.showerror('Invalid session options',str(exc))

    def run():
        if not session: return
        current = session
        def analyze():
            result = recognize(current)
            auto = simulation_completion(result,simulation_info(current))
            export_attendance(current,result,**auto)
            return result
        submit(analyze, lambda _: display(current))

    def save():
        if not session: return
        if future is not None and not future.done():
            status.set('Attendez la fin de la reconnaissance avant de sauvegarder la présence.'); return
        try:
            result = read_json(session_path(session)/'predictions.json')
            table = export_attendance(session,result,completed.get(),coverage.get())
            show_table(table); status.set(f'CSV sauvegardés : présence complète, absents, à vérifier et non reconnus — {session}')
        except Exception as exc: messagebox.showerror('Save failed',str(exc))

    def compare():
        if not session: return
        try:
            report = compare_truth(session)
            lines = [f"{r['truth']}: {r['prediction']}" for r in report['matches']]
            messagebox.showinfo('Simulation comparison', '\n'.join(lines)+f"\nExtra detections: {report['extra_detections']}")
        except Exception as exc: messagebox.showerror('Run recognition first',str(exc))

    def navigate(delta):
        if future is not None and not future.done(): return
        sessions = sorted(p.name for p in (ROOT/'sessions').glob('*') if p.is_dir() and (p/'classroom.png').exists())
        if sessions:
            i = sessions.index(session) if session in sessions else -1
            display(sessions[(i+delta)%len(sessions)])

    def inspect_faces():
        if not session: return
        try:
            result = read_json(session_path(session)/'predictions.json')
            lines = []
            for i, face in enumerate(result['faces'],1):
                distance = face.get('distance')
                text = f"Face {i}: {face['name']}"
                if distance is not None:
                    text += f"\nClosest: {face.get('best_candidate','n/a')} | distance {distance:.3f}"
                    if 'threshold' in face: text += f" | acceptance threshold {face['threshold']:.3f}"
                    if face.get('margin') is not None:
                        text += f"\nMargin {face['margin']:.3f} | required {face.get('required_margin',0):.3f}"
                text += '\n' + face.get('reason','')
                if face.get('candidates'): text += '\nCandidates: ' + ', '.join(face['candidates'])
                lines.append(text)
            messagebox.showinfo('Face inspection', '\n\n'.join(lines) if lines else 'No faces detected.')
        except Exception as exc: messagebox.showerror('Run recognition first',str(exc))

    for text, command in [('Generate',new),('Recognize',run),('Save CSV',save),('Compare truth',compare),('Previous',lambda:navigate(-1)),('Next',lambda:navigate(1)),('Inspect',inspect_faces)]:
        ttk.Button(controls,text=text,command=command).pack(side='left',padx=3)
    def close():
        pool.shutdown(wait=False, cancel_futures=True); window.destroy()
    window.protocol('WM_DELETE_WINDOW',close)
    window.mainloop()
