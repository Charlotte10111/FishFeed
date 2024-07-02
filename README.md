gcloud builds submit --tag gcr.io/final-year-project-427106/fish-feeding --project=final-year-project-427106
gcloud run deploy --image gcr.io/final-year-project-427106/fish-feeding --platform managed  --project=final-year-project-427106 --allow-unauthenticated
