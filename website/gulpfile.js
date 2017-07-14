var gulp = require('gulp');
var sass = require('gulp-sass');
var cleanCSS = require('gulp-clean-css');
var rename = require('gulp-rename');

gulp.task('sass', function() {
    return gulp.src('assets/themes/mahout4/scss/style.scss')
        .pipe(sass())
        .pipe(gulp.dest('assets/themes/mahout4/css'))
});

gulp.task('css-min', ['sass'], function() {
    return gulp.src('assets/themes/mahout4/css/style.css')
        .pipe(cleanCSS({ compatibility: 'ie8' }))
        .pipe(rename({ suffix: '.min' }))
        .pipe(gulp.dest('assets/themes/mahout4/css'))
});

gulp.task('default', ['sass', 'css-min']);

gulp.task('dev', ['sass', 'css-min'], function() {
    gulp.watch('assets/themes/mahout4/scss/style.scss', ['sass']);
    gulp.watch('assets/themes/mahout4/css/style.css', ['css-min']);
});
