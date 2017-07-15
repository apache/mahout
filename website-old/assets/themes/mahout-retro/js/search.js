function initSearch(){
    var methods = {
        defaultValueActsAsHint: function(element){
            element = $(element);
            element._default = element.value;
            return element.observe('focus', function(){
                if(element._default != element.value) return;
                element.removeClassName('hint').value = '';
            }).observe('blur', function(){
                if(element.value.strip() != '') return;
                element.addClassName('hint').value = element._default;
            }).addClassName('hint');
        }
    };
    $w('input textarea').each(function(tag){ Element.addMethods(tag, methods) });
}
initSearch();

document.observe('dom:loaded', function(){
    $('searchDocs').defaultValueActsAsHint();
});