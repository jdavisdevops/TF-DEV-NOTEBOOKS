WITH gpa_calc AS (
    SELECT
        st.id,
        CASE
            WHEN st.grade_level < 9    THEN
                (
                    SELECT
                        round(SUM(sg.gpa_points) / COUNT(sg.course_number), 2)
                    FROM
                        storedgrades sg
                    WHERE
                            sg.studentid = st.id
                        AND sg.grade_level IN (
                            6,
                            7,
                            8
                        )
                )
            WHEN st.grade_level > 8    THEN
                (
                    SELECT
                        round(SUM(sg.gpa_points) / COUNT(sg.course_number), 2)
                    FROM
                        storedgrades sg
                    WHERE
                            sg.studentid = st.id
                        AND sg.grade_level > 8
                        AND sg.excludefromgpa <> 1
                )
        END AS gpa
    FROM
        students st
    WHERE
        st.enroll_status = 0
)
SELECT
    to_char(s.student_number)                                                                                                                                               AS student_number,
    s.lastfirst,
    s.grade_level,
    (
        SELECT
            x.elastatus
        FROM
            s_ca_stu_x x
        WHERE
            x.studentsdcid = s.dcid
    ) AS "ELA Status",
    (
        SELECT
            CASE
                WHEN x.primarydisability IS NOT NULL THEN
                    'Y'
                ELSE
                    'N'
            END AS "SPECIAL_ED_STUDENT"
        FROM
            s_ca_stu_x x
        WHERE
            x.studentsdcid = s.dcid
    ) AS "SPED Status",
    CASE
        WHEN to_char(gpa_calc.gpa) IS NULL THEN
            to_char('NA')
        ELSE
            to_char(gpa_calc.gpa)
    END AS gpa,
    pgfg.grade,
    pgfg.citizenship,
    cc.currentabsences,
    cc.currenttardies,
    cal.programcode
FROM
    students       s,
    pgfinalgrades  pgfg,
    sections       sec,
    courses        c,
    cc,
    S_CA_STU_CALPADSPrograms_C cal,
    gpa_calc
WHERE
    ( ps_customfields.getstudentscf(s.id, 'AUSD_InstrSet') <> 'C'
      OR ps_customfields.getstudentscf(s.id, 'AUSD_InstrSet') IS NULL )
    AND s.id = pgfg.studentid
--    AND pgfg.finalgradename in ('Q4','S2')
    AND s.id = gpa_calc.id
    AND sec.id = pgfg.sectionid
    AND c.course_number = sec.course_number
    and s.dcid = cal.studentsdcid
    and cal.programcode = '122'
    AND abs(cc.sectionid) = pgfg.sectionid
    AND cc.studentid = pgfg.studentid
--    AND cc.termid LIKE 30 || '%%'
    and s.enroll_status = 0
    AND pgfg.grade <> '--'
    AND pgfg.grade IS NOT NULL
    order by student_number