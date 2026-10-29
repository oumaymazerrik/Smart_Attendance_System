from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
from .common import ROOT, roster, write_json, read_json


def simulation_info(session_id):
    path = ROOT/'sessions'/session_id/'simulation.json'
    if not path.exists(): return None
    info = read_json(path)
    registered, generated = info.get('registered_count'), info.get('generated_count')
    if (info.get('kind') != 'synthetic_classroom' or info.get('session_id') != session_id
        or type(registered) is not int or type(generated) is not int
        or not 0 <= generated <= registered):
        return None
    return {**info,'expected_absent_count':registered-generated}


def simulation_completion(result, info):
    """Generated still images are complete; compare detection counts, never identities.

    Equal counts provide a simulation coverage check, not a guarantee for arbitrary
    imported images. Unknown/ambiguous predictions still follow conservative rules.
    """
    adequate = bool(info and info['registered_count']==len(roster()) and not result.get('errors')
                    and result.get('detected_count')==info['generated_count']
                    and len(result.get('faces',[]))==info['generated_count'])
    return {'completed':bool(info),'coverage_adequate':adequate}


def attendance_table(predictions, completed=False, coverage_adequate=False, timestamp=None):
    now = datetime.fromisoformat(timestamp) if timestamp else datetime.now(ZoneInfo('Africa/Casablanca'))
    recognized, review = {}, {}
    registered = {r['Name'] for r in roster()}
    for face in predictions:
        name, distance = face['name'], face.get('distance')
        if name in registered:
            recognized.setdefault(name, []).append(distance)
        elif name == 'Needs Review':
            for candidate in face['candidates']:
                if candidate in registered: review[candidate] = distance
    unresolved = any(face['name'] == 'Unknown' for face in predictions) or any(len(v) > 1 for v in recognized.values())
    rows = []
    for student in roster():
        name = student['Name']
        if len(recognized.get(name, [])) == 1:
            status, distance = 'Present', recognized[name][0]
        elif name in recognized:  # Multiple visible faces assigned to one identity need inspection.
            status, distance = 'Needs Review', min(recognized[name])
        elif name in review:
            status, distance = 'Needs Review', review[name]
        else:
            status, distance = ('Absent' if completed and coverage_adequate and not unresolved else 'Needs Review'), None
        rows.append({**student, 'Date': now.strftime('%Y-%m-%d'), 'Time': now.strftime('%H:%M:%S'),
                     'Status': status, 'Distance': distance})
    return pd.DataFrame(rows, columns=['Student_ID','Name','Date','Time','Status','Distance'])


def attendance_groups(table):
    """Keep confirmed absences separate from recognition uncertainty."""
    return {
        'present': table.loc[table.Status.eq('Present')].copy(),
        'absent': table.loc[table.Status.eq('Absent')].copy(),
        'review': table.loc[table.Status.eq('Needs Review')].copy(),
        'not_recognized': table.loc[~table.Status.eq('Present')].copy(),
    }


def export_attendance(session_id, result, completed=False, coverage_adequate=False):
    if not session_id or Path(session_id).name != session_id:
        raise ValueError('Invalid session ID')
    if result.get('errors'): coverage_adequate = False
    table = attendance_table(result['faces'], completed, coverage_adequate, result['timestamp'])
    target = ROOT / 'attendance_reports' / f'{session_id}.csv'
    target.parent.mkdir(exist_ok=True)
    table.to_csv(target, index=False)
    groups = attendance_groups(table)
    for group, suffix in (('absent','absents'),('review','a_verifier'),('not_recognized','non_reconnus')):
        groups[group].to_csv(target.with_name(f'{session_id}_{suffix}.csv'),index=False)
    policy = {'completed':bool(completed),'coverage_adequate':bool(coverage_adequate)}
    write_json(target.with_suffix('.json'),{'session_id':session_id,**policy,
               'counts':{name:len(rows) for name,rows in groups.items()},
               'simulation':simulation_info(session_id),
               'note':'Non-reconnus = absents confirmés + personnes à vérifier; ce n’est pas une liste d’absents confirmés.'})
    session_directory = ROOT/'sessions'/session_id
    if session_directory.is_dir(): write_json(session_directory/'attendance_policy.json',policy)
    return table
