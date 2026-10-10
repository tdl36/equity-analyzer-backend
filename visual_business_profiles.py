"""Public business context, never private research. Curated comparisons are not rankings."""
PROFILES = {
 'UNH': {
  'name':'UnitedHealth Group', 'industry':'Managed care & health services',
  'source':'https://www.unitedhealthgroup.com/content/dam/UHG/PDF/investors/2025/UNH-Q4-2025-Form-10-K.pdf',
  'basis':'FY2025 operating structure; Optum Financial moved from Health to Insight in 2026.',
  'overview':'Health benefits through UnitedHealthcare; care delivery, technology and pharmacy services through Optum.',
  'segments':{
   'UnitedHealthcare':('Health benefits','Premiums and service fees','Medical costs, membership','insurance'),
   'Optum Health':('Care delivery','Patient care and service revenue','Patients served, care margins','care'),
   'Optum Insight':('Technology & analytics','Software and services','Backlog, execution','monitor'),
   'Optum Rx':('Pharmacy services','Products and pharmacy services','Prescriptions, drug mix','pharmacy'),
  },
  'peers':[('CVS','CVS Health','Benefits + pharmacy'),('CI','The Cigna Group','Benefits + services'),('HUM','Humana','Medicare-focused')],
  'peerBasis':'Selected US health-benefit comparables; business mixes differ. No ranking or pure-play equivalence implied.',
 }
}
