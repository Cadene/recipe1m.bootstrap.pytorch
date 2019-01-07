$(document).ready(function () {

    // Image processing

    $(document).on('change', '.btn-file :file', function() {
        var input = $(this),
            label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
        input.trigger('fileselect', [label]);
    });

    $('.btn-file :file').on('fileselect', function(event, label) {
        var input = $(this).parents('.input-group').find(':text'),
            log = label;
        
        if( input.length ) {
            input.val(log);
        } else {
            if( log ) alert(log);
        }
    
    });

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            
            reader.onload = function (e) {
                $('#adamine-image').attr('src', e.target.result);
            }
            
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imgInp").change(function(){
        readURL(this);
    });     

    // Send Image

    $(document).ajaxStart(function () {
        $('#loading').show();
        $('#adamine-recipe').hide();
    }).ajaxStop(function () {
        $('#loading').hide();
        $('#adamine-recipe').show();
    });

    var formBasic = function () {
        var formData = $("#formBasic").serialize();
        var data = {
            image: $('#adamine-image').attr('src'),
            mode: 'all'
        }
                     //question : $('#adamine-question').val()}
        $.ajax({

            type: 'post',
            data: data,
            dataType: 'json',
            url: 'http://edwards:3456', // your global ip address and port

            error: function () {
                alert("There was an error processing this page.");
                return false;
            },

            complete: function (output) {
                if ('responseJSON' in output) {
                    var ul = $('<ul></ul>');
                    
                    for (i=0; i < output.responseJSON.length; i++) {
                        var li = $('<li></li>');
                        var out = output.responseJSON[i]

                        li.append($('<p>class: ' + out['class_name'] + '</p>'));
                        li.append($('<p>title: <a href="' + out['url'] + '">' + out['title'] + '</a></p>'));
                        li.append($('<img width="200px" src="' + out['img_strb64'] + '"/>'));

                        ingrs = []
                        for (j=0; j < out['ingredients'].length; j++) {
                            ingrs.push(out['ingredients'][j]['text'])
                        }
                        li.append($('<p>ingredients: ' + ingrs.join(', ') + '</p>'));

                        instrs = []
                        for (j=0; j < out['instructions'].length; j++) {
                            instrs.push(out['instructions'][j]['text'])
                        }
                        li.append($('<p>instructions: ' + instrs.join('\n') + '</p>'));

                        ul.append(li);
                    }

                    $('#adamine-recipe').html(ul);
                    console.log(output.responseJSON);

                } else if ('responseText' in output) {
                    alert(output.responseText);
                    console.log(output.responseText);

                } else {
                    alert('Something wrong happend!');
                    console.log(output)
                }
            }
        });
        return false;
    };

   $("#basic-submit").on("click", function (e) {
       e.preventDefault();
       formBasic();
    });
});